#include <ap_int.h>
#include <hls_stream.h>
#include <ap_axi_sdata.h>
#include <math.h>
#include "../src/stencil.h"
#include <stdio.h>
#include "../src/stencil.cpp"


// coalesced memory access at 512 bit to get maximum out of memory bandwidth
// Single pipelined loop below will be mapped to single memory transfer
// which will further split into multiple transfers by axim module.
static void read_tile(uint512_dt*  arg0, hls::stream<uint512_dt> &rd_buffer, struct data_G data_g){

	unsigned int total_itr = data_g.total_itr;
	for(unsigned int itr = 0; itr < total_itr; itr++){
		#pragma HLS PIPELINE II=1
		rd_buffer << arg0[itr];
	}

}


// data width conversion from 512 bit to 256 bit width of
// compute pipeline
static void stream_convert_512_256(hls::stream<uint512_dt> &in, hls::stream<uint256_dt> &out, struct data_G data_g){
	unsigned int total_itr = data_g.total_itr;
	for(unsigned int itr = 0; itr < total_itr; itr++){
			#pragma HLS PIPELINE II=2
			uint512_dt tmp = in.read();
			uint256_dt var_l = tmp.range(255,0);
			uint256_dt var_h = tmp.range(511,256);;
			out << var_l;
			if(!data_g.last_half ||  itr < total_itr -1){
				out << var_h;
			}
	}
}

// data width conversion to support 512 bit width memory write interface
static void stream_convert_256_512(hls::stream<uint256_dt> &in, hls::stream<uint512_dt> &out, struct data_G data_g){
	unsigned int total_itr = data_g.total_itr;
	for(unsigned int itr = 0; itr < total_itr; itr++){
		#pragma HLS PIPELINE II=2
		uint512_dt tmp;
		tmp.range(255,0) = in.read();
		if(!data_g.last_half ||  itr < total_itr -1){
			tmp.range(511,256) = in.read();
		}
		out << tmp;
	}
}

// coalesced memory write using 512 bit to get maximum out of memory bandwidth
// Single pipelined loop below will be mapped to single memory transfer
// which will further split into multiple transfers by axim module.
static void write_tile(uint512_dt*  arg1, hls::stream<uint512_dt> &wr_buffer, struct data_G data_g){
	unsigned int total_itr = data_g.total_itr;
	for(unsigned int itr = 0; itr < total_itr; itr++){
		 #pragma HLS PIPELINE II=2
		 arg1[itr] = wr_buffer.read();
	}
}

static void process_ReadWrite (uint512_dt*  arg0_0, uint512_dt*  arg1_0,
				   hls::stream <t_pkt> &in, hls::stream <t_pkt> &out,
				   const int xdim0, const unsigned short size_x, const unsigned short size_y, const unsigned short size_z, const unsigned short batches){


    static hls::stream<uint256_dt> streamArray[40 + 1];
    static hls::stream<uint512_dt> rd_bufferArr[4];
    static hls::stream<uint512_dt> wr_bufferArr[4];
    static hls::stream<uint512_dt> streamArray_512[4];

    #pragma HLS STREAM variable = streamArray depth = 2
	#pragma HLS STREAM variable = streamArray_512 depth = 2
	#pragma HLS STREAM variable = rd_bufferArr depth = max_depth_16
	#pragma HLS STREAM variable = wr_bufferArr depth = max_depth_16



    struct data_G data_g;
    data_g.sizex = size_x;
    data_g.sizey = size_y;
    data_g.sizez = size_z;
    data_g.xdim = xdim0;
	data_g.xblocks = (xdim0 >> (SHIFT_BITS+1));
	data_g.grid_sizey = size_y+2;
	data_g.grid_sizez = size_z+2;
	data_g.limit_z = size_z+3;

	data_g.offset_x = 0;
	data_g.tile_x = xdim0;
	data_g.offset_y = 0;
	data_g.tile_y = size_y+2;

	data_g.plane_size = data_g.xblocks * data_g.grid_sizey;

	unsigned int tile_plane_size = (data_g.tile_x >> SHIFT_BITS) * data_g.tile_y;
	unsigned int totol_iter = register_it<unsigned int>(tile_plane_size * data_g.grid_sizez) * batches;

	data_g.last_half = totol_iter & 0x1;
	data_g.total_itr = ((totol_iter+1) >> 1);


	#pragma HLS dataflow

	read_tile(arg0_0, rd_bufferArr[0], data_g);
	stream_convert_512_256(rd_bufferArr[0], streamArray[0], data_g);

	// sending data out for compute kernel
	fifo256_2axis(streamArray[0], out, totol_iter);
	// receiving data from compute kernel
	axis2_fifo256(in, streamArray[31], totol_iter);

	stream_convert_256_512(streamArray[31],wr_bufferArr[0], data_g);
	write_tile(arg1_0, wr_bufferArr[0], data_g);
}




static void process_ReadWrite_dataflow (uint512_dt*  arg0_0, uint512_dt*  arg1_0,
				   hls::stream <t_pkt> &in, hls::stream <t_pkt> &out,
				   const int xdim0, const unsigned short size_x, const unsigned short size_y, const unsigned short size_z, const unsigned short batches){

		#pragma HLS dataflow
		process_ReadWrite(arg0_0, arg1_0, in, out, xdim0, size_x, size_y, size_z, batches);


}

//-DHOST_CODE_OPT -DLOCAL_BUF_OPT -DDF_OPT -DFP_OPT


// kernel for reading and writing from memory
// it resides in SLR0 which is close to HBM memory and DDR4[0]

extern "C" {
void stencil_Read_Write(
		uint512_dt*  arg0_0,
		uint512_dt*  arg1_0,


		const int sizex,
		const int sizey,
		const int sizez,
		const int xdim0,
		const int batches,
		const int count,

		hls::stream <t_pkt> &in,
		hls::stream <t_pkt> &out){

	#pragma HLS INTERFACE depth=4096 m_axi port = arg0_0 offset = slave bundle = gmem0 max_read_burst_length=64 max_write_burst_length=64 num_read_outstanding=4 num_write_outstanding=4 latency=64
	#pragma HLS INTERFACE depth=4096 m_axi port = arg1_0 offset = slave bundle = gmem0


	#pragma HLS INTERFACE s_axilite port = arg0_0 bundle = control
	#pragma HLS INTERFACE s_axilite port = arg1_0 bundle = control


	#pragma HLS INTERFACE axis port = in  register
	#pragma HLS INTERFACE axis port = out register

	#pragma HLS INTERFACE s_axilite port = batches bundle = control
	#pragma HLS INTERFACE s_axilite port = sizex bundle = control
	#pragma HLS INTERFACE s_axilite port = sizey bundle = control
	#pragma HLS INTERFACE s_axilite port = sizez bundle = control
	#pragma HLS INTERFACE s_axilite port = xdim0 bundle = control
	#pragma HLS INTERFACE s_axilite port = count bundle = control
	#pragma HLS INTERFACE s_axilite port = return bundle = control


	unsigned short count_s = count;
	for(unsigned short i =  0; i < count_s; i++){
		process_ReadWrite_dataflow(arg0_0, arg1_0, in, out, xdim0, sizex, sizey, sizez, batches);
		process_ReadWrite_dataflow(arg1_0, arg0_0, in, out, xdim0, sizex, sizey, sizez, batches);
	}
}
}
