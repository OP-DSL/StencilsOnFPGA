#include <ap_int.h>
#include <hls_stream.h>
#include <ap_axi_sdata.h>
#include <math.h>
#include "../src/stencil.h"
#include <stdio.h>




static void read_tile(uint512_dt*  arg0, hls::stream<uint512_dt> &rd_buffer, const int xdim0, const int offset, int tile_x, int size_y, unsigned short start){
	unsigned short end_index = (tile_x >> (SHIFT_BITS+1));
	unsigned short  end_row = size_y+2;
	for(unsigned short i = start; i < end_row; i = i+2){
		unsigned int base_index = (offset + xdim0*i) >> (SHIFT_BITS+1);
		for (unsigned short j = 0; j < end_index; j++){
			#pragma HLS PIPELINE II=1
			rd_buffer << arg0[base_index + j];
		}
	}
}

static void combine_tile(hls::stream<uint512_dt> &rd_buffer0, hls::stream<uint512_dt> &rd_buffer1, hls::stream<uint512_dt> &rd_buffer2, hls::stream<uint512_dt> &rd_buffer3,
		hls::stream<uint512_dt> &combined_buffer, const int xdim0, const int offset, int tile_x, int size_y){
	unsigned short end_index = (tile_x >> (SHIFT_BITS+1));
	unsigned short  end_row = size_y+2;
	for(unsigned short i = 0; i < end_row; i = i+2){
		for (unsigned short j = 0; j < end_index; j++){
			#pragma HLS PIPELINE II=1
			combined_buffer << rd_buffer0.read();
		}

		for (unsigned short j = 0; j < end_index; j++){
			#pragma HLS PIPELINE II=1
			combined_buffer << rd_buffer1.read();
		}

//		for (unsigned short j = 0; j < end_index; j++){
//			#pragma HLS PIPELINE II=1
//			combined_buffer << rd_buffer2.read();
//		}
//
//		for (unsigned short j = 0; j < end_index; j++){
//			#pragma HLS PIPELINE II=1
//			combined_buffer << rd_buffer3.read();
//		}
	}
}


static void stream_convert_512_256(hls::stream<uint512_dt> &in, hls::stream<uint256_dt> &out, const int tile_x, int size_y){
	unsigned short end_index = (tile_x >> (SHIFT_BITS+1));
	unsigned short end_row = size_y+2;
	unsigned int tot_itr = end_row * end_index;
	for (int itr = 0; itr < tot_itr; itr++){
		#pragma HLS PIPELINE II=2
		#pragma HLS loop_tripcount min=min_grid max=max_grid avg=avg_grid
		uint512_dt tmp = in.read();
		uint256_dt var_l = tmp.range(255,0);
		uint256_dt var_h = tmp.range(511,256);;
		out << var_l;
		out << var_h;
	}
}

static void stream_convert_256_512(hls::stream<uint256_dt> &in, hls::stream<uint512_dt> &out, const int tile_x, int size_y, unsigned short offset){
	unsigned short adjust = (offset == 0)? 0: 2;
	unsigned short end_index = (tile_x >> (SHIFT_BITS+1));
	unsigned short end_row = size_y+2;
	unsigned int tot_itr = end_row * end_index;
	unsigned short j = 0;
	for (int itr = 0; itr < tot_itr; itr++){
		#pragma HLS PIPELINE II=2
		#pragma HLS loop_tripcount min=min_grid max=max_grid avg=avg_grid
		uint512_dt tmp;
		uint256_dt tmp0_in = in.read();
		uint256_dt tmp1_in = in.read();
		bool cond = (j == end_index);
		if(cond){
			j = 0;
		}
		tmp.range(255,0) = tmp0_in;
		tmp.range(511,256) = tmp1_in;
		bool skip = offset != 0 && (j == 0 || j == 1);
		if(!skip){
			out << tmp;
		}
		j++;
	}
}

static void axis2_fifo256(hls::stream <t_pkt> &in, hls::stream<uint256_dt> &out, const int tile_x, int size_y){
	unsigned short end_index = (tile_x >> (SHIFT_BITS));
	unsigned short end_row = size_y+2;
	unsigned int tot_itr = end_row * end_index;
	for (int itr = 0; itr < tot_itr; itr++){
		#pragma HLS PIPELINE II=1
		#pragma HLS loop_tripcount min=min_grid max=max_grid avg=avg_grid
		t_pkt tmp = in.read();
		out << tmp.data;
	}
}

static void fifo256_2axis(hls::stream <uint256_dt> &in, hls::stream<t_pkt> &out, const int tile_x, int size_y){
	unsigned short end_index = (tile_x >> (SHIFT_BITS));
	unsigned short end_row = size_y+2;
	unsigned int tot_itr = end_row * end_index;
	for (int itr = 0; itr < tot_itr; itr++){
		#pragma HLS PIPELINE II=1
		#pragma HLS loop_tripcount min=min_grid max=max_grid avg=avg_grid
		t_pkt tmp;
		tmp.data = in.read();
		out.write(tmp);
	}
}



static void separate_tile(hls::stream<uint512_dt> &in_buffer, hls::stream<uint512_dt> &wr_buffer0, hls::stream<uint512_dt> &wr_buffer1, hls::stream<uint512_dt> &wr_buffer2,
		hls::stream<uint512_dt> &wr_buffer3, const int xdim0, const int offset, int tile_x, int size_y){
	unsigned short full_size = (tile_x >> (SHIFT_BITS+1));
	unsigned short adjust = offset == 0 ? 0 : 2;
	unsigned short end_index = full_size-adjust;
	unsigned short end_row = size_y + 2;
	for(unsigned short i = 0; i < end_row; i = i+2){
		for (unsigned short j = 0; j < end_index; j++){
			#pragma HLS PIPELINE II=1
			wr_buffer0 << in_buffer.read();
		}

		for (unsigned short j = 0; j < end_index; j++){
			#pragma HLS PIPELINE II=1
			wr_buffer1 << in_buffer.read();
		}
//
//		for (unsigned short j = 0; j < end_index; j++){
//			#pragma HLS PIPELINE II=1
//			wr_buffer2 << in_buffer.read();
//		}
//
//		for (unsigned short j = 0; j < end_index; j++){
//			#pragma HLS PIPELINE II=1
//			wr_buffer3 << in_buffer.read();
//		}
	}
}

static void write_tile( uint512_dt*  arg1, hls::stream<uint512_dt> &wr_buffer, const int xdim0, const unsigned short offset,
			unsigned short tile_x, unsigned short size_y, unsigned short start){
	unsigned short full_size = (tile_x >> (SHIFT_BITS+1));
	unsigned short adjust = offset == 0 ? 0 : 2;
	unsigned short end_index = full_size-adjust;
	unsigned short end_row = size_y+2;
	for(unsigned short i = start; i < end_row; i = i + 2){
		unsigned int  base_index = ((offset + xdim0*i) >> (SHIFT_BITS+1)) + adjust;
		for (unsigned short j = 0; j < end_index; j++){
			#pragma HLS PIPELINE II=1
			arg1[base_index + j] = wr_buffer.read();
		}
	}
}


static void process_ReadWrite (uint512_dt*  arg0_0, uint512_dt*  arg0_1,
				   uint512_dt*  arg1_0,  uint512_dt*  arg1_1,
				   hls::stream <t_pkt> &in, hls::stream <t_pkt> &out,
				   const int xdim0, const unsigned short offset, const unsigned short tile_x, const unsigned short size_x, const unsigned short size_y){


    static hls::stream<uint256_dt> streamArray[40 + 1];
    static hls::stream<uint512_dt> rd_bufferArr[4];
    static hls::stream<uint512_dt> wr_bufferArr[4];
    static hls::stream<uint512_dt> streamArray_512[4];

    #pragma HLS STREAM variable = streamArray depth = 2
	#pragma HLS STREAM variable = streamArray_512 depth = 2
	#pragma HLS STREAM variable = rd_bufferArr depth = max_depth_16
	#pragma HLS STREAM variable = wr_bufferArr depth = max_depth_16

	#pragma HLS dataflow


	struct data_G data_g;
	data_g.end_index = (tile_x >> SHIFT_BITS);
	data_g.end_row = size_y+2;
	data_g.outer_loop_limit = size_y+3;
	data_g.gridsize = (data_g.outer_loop_limit * data_g.end_index +1);
	data_g.endindex_minus1 = data_g.end_index -1;
	data_g.endrow_plus1 = data_g.end_row + 1;
	data_g.endrow_plus2 = data_g.end_row + 2;
	data_g.endrow_minus1 = data_g.end_row - 1;





	read_tile(arg0_0, rd_bufferArr[0], xdim0, offset, tile_x, size_y, 0);
	read_tile(arg0_1, rd_bufferArr[1], xdim0, offset, tile_x, size_y, 1);


	combine_tile(rd_bufferArr[0], rd_bufferArr[1], rd_bufferArr[2], rd_bufferArr[3], streamArray_512[0], xdim0, offset, tile_x, size_y);
	stream_convert_512_256(streamArray_512[0], streamArray[0], tile_x, size_y);


	fifo256_2axis(streamArray[0], out, tile_x, size_y);
	axis2_fifo256(in, streamArray[31], tile_x, size_y);



	stream_convert_256_512(streamArray[31], streamArray_512[1], tile_x, size_y, offset);

	separate_tile(streamArray_512[1], wr_bufferArr[0], wr_bufferArr[1], wr_bufferArr[2], wr_bufferArr[3], xdim0, offset, tile_x, size_y);
	write_tile(arg1_0, wr_bufferArr[0], xdim0, offset, tile_x, size_y, 0);
	write_tile(arg1_1, wr_bufferArr[1], xdim0, offset, tile_x, size_y, 1);


}




static void process_ReadWrite_dataflow (uint512_dt*  arg0_0, uint512_dt*  arg0_1,
				   uint512_dt*  arg1_0,  uint512_dt*  arg1_1,
				   hls::stream <t_pkt> &in, hls::stream <t_pkt> &out,
				   const int xdim0, unsigned int tile_mem[], unsigned short tile_count , const unsigned short size_x, const unsigned short size_y){

	for(int j = 0; j < tile_count; j++){
			#pragma HLS dataflow
			unsigned short offset = tile_mem[j] & 0xffff;
			unsigned short tile_x = tile_mem[j] >> 16;
			process_ReadWrite(arg0_0, arg0_1, arg1_0, arg1_1, in, out, xdim0, offset, tile_x, size_x, size_y);
		}

}

//-DHOST_CODE_OPT -DLOCAL_BUF_OPT -DDF_OPT -DFP_OPT


extern "C" {
void stencil_Read_Write(
		uint512_dt*  arg0_0,
		uint512_dt*  arg0_1,

		uint512_dt*  arg1_0,
		uint512_dt*  arg1_1,

		const unsigned int* tile,
		const int tile_count,
		const int sizex,
		const int sizey,
		const int xdim0,
		const int count,

		hls::stream <t_pkt_32> &tile_s_out,
		hls::stream <t_pkt> &in,
		hls::stream <t_pkt> &out){

	#pragma HLS INTERFACE depth=4096 m_axi port = arg0_0 offset = slave bundle = gmem0 max_read_burst_length=64 max_write_burst_length=64 num_read_outstanding=2 num_write_outstanding=2
	#pragma HLS INTERFACE depth=4096 m_axi port = arg0_1 offset = slave bundle = gmem1 max_read_burst_length=64 max_write_burst_length=64 num_read_outstanding=2 num_write_outstanding=2


	#pragma HLS INTERFACE depth=4096 m_axi port = arg1_0 offset = slave bundle = gmem0
	#pragma HLS INTERFACE depth=4096 m_axi port = arg1_1 offset = slave bundle = gmem1

	#pragma HLS INTERFACE depth=4096 m_axi port = tile offset = slave bundle = gmem4

	#pragma HLS INTERFACE s_axilite port = arg0_0 bundle = control
	#pragma HLS INTERFACE s_axilite port = arg0_1 bundle = control

	#pragma HLS INTERFACE s_axilite port = arg1_0 bundle = control
	#pragma HLS INTERFACE s_axilite port = arg1_1 bundle = control


	#pragma HLS INTERFACE s_axilite port = tile bundle = control
	#pragma HLS INTERFACE axis port = tile_s_out  register
	#pragma HLS INTERFACE axis port = in  register
	#pragma HLS INTERFACE axis port = out register

	#pragma HLS INTERFACE s_axilite port = tile_count bundle = control
	#pragma HLS INTERFACE s_axilite port = sizex bundle = control
	#pragma HLS INTERFACE s_axilite port = sizey bundle = control
	#pragma HLS INTERFACE s_axilite port = xdim0 bundle = control
	#pragma HLS INTERFACE s_axilite port = count bundle = control
	#pragma HLS INTERFACE s_axilite port = return bundle = control



	unsigned int tile_mem[256];

	for(int j = 0; j < tile_count; j++){
		#pragma HLS PIPELINE II=1
		t_pkt_32 tmp_s;
		unsigned int tmp;
		tmp = tile[j];
		tile_mem[j] = tmp;
		tmp_s.data = tmp;
		tile_s_out.write(tmp_s);
	}



	bool read_write_offset_dn = 0;
	for(int i =  0; i < count; i++){
//		for(int j = 0; j < tile_count; j++){
//			//#pragma HLS dataflow
//			unsigned short offset = tile_mem[j] & 0xffff;
//			unsigned short tile_x = tile_mem[j] >> 16;
//			process_ReadWrite(arg0_0, arg0_1, arg1_0, arg1_1, in, out, xdim0, offset, tile_x, sizex, sizey);
//		}
//
//		for(int j = 0; j < tile_count; j++){
//			//#pragma HLS dataflow
//			unsigned short offset = tile_mem[j] & 0xffff;
//			unsigned short tile_x = tile_mem[j] >> 16;
//			process_ReadWrite(arg1_0, arg1_1, arg0_0, arg0_1, in, out, xdim0, offset, tile_x, sizex, sizey);
//		}

		process_ReadWrite_dataflow(arg0_0, arg0_1, arg1_0, arg1_1, in, out, xdim0, tile_mem, tile_count, sizex, sizey);
		process_ReadWrite_dataflow(arg1_0, arg1_1, arg0_0, arg0_1, in, out, xdim0, tile_mem, tile_count, sizex, sizey);
	}
}
}
