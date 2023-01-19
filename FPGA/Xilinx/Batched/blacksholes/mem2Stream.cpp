#include <ap_int.h>
#include <hls_stream.h>
#include <ap_axi_sdata.h>
#include <math.h>
#include <stdio.h>
#include "blacksholes_common.h"
#include "stencil.h"
#include "stencil.cpp"

// coalesced memory access at 512 bit to get maximum out of memory bandwidth
// Single pipelined loop below will be mapped to single memory transfer
// which will further split into multiple transfers by axim module.
static void read_grid(uint512_dt*  arg0, hls::stream<uint512_dt> &rd_buffer, const unsigned int total_itr)
{
	for (int itr = 0; itr < total_itr; itr++){
		#pragma HLS PIPELINE II=1
		#pragma HLS loop_tripcount min=min_grid max=max_grid avg=avg_grid
		rd_buffer << arg0[itr];
	}
}

// data width conversion to support 256 bit width compute pipeline
static void stream_convert_512_256(hls::stream<uint512_dt> &in, hls::stream<uint256_dt> &out,
		const unsigned int total_itr_512, const unsigned int total_iter_256)
{
	unsigned int total_itr = total_itr_512;
	bool flag = total_iter_256 & 0x1;

	for (int itr = 0; itr < total_itr; itr++){
		#pragma HLS PIPELINE II=2
		#pragma HLS loop_tripcount min=min_grid max=max_grid avg=avg_grid
		uint512_dt tmp = in.read();
		uint256_dt var_l = tmp.range(255,0);
		uint256_dt var_h = tmp.range(511,256);;
		out << var_l;
		if(~flag ||  itr < total_itr -1){
			out << var_h;
		}
	}
}

// data width conversion to support 512 bit width memory write interface
static void stream_convert_256_512(hls::stream<uint256_dt> &in, hls::stream<uint512_dt> &out,
		const unsigned int total_itr_512, const unsigned int total_itr_256)
{
	unsigned int total_itr = total_itr_512;
	bool flag = total_itr_256 & 0x1;
	for (int itr = 0; itr < total_itr; itr++){
		#pragma HLS PIPELINE II=2
		#pragma HLS loop_tripcount min=min_grid max=max_grid avg=avg_grid
		uint512_dt tmp;
		tmp.range(255,0) = in.read();
		if(~flag ||  itr < total_itr -1){
			tmp.range(511,256) = in.read();
		}
		out << tmp;
	}
}

// coalesced memory write using 512 bit to get maximum out of memory bandwidth
// Single pipelined loop below will be mapped to single memory transfer
// which will further split into multiple transfers by axim module.
static void write_grid(uint512_dt*  arg0, hls::stream<uint512_dt> &wr_buffer, const unsigned int total_itr)
{
	for (int itr = 0; itr < total_itr; itr++){
		#pragma HLS PIPELINE II=1
		#pragma HLS loop_tripcount min=min_grid max=max_grid avg=avg_grid
		arg0[itr] = wr_buffer.read();
	}
}


void process_mem2stream(uint512_dt* arg0, uint512_dt* arg1, const int count, const int xdim0, const int ydim0,
			const int batch, hls::stream <t_pkt> &in, hls::stream <t_pkt> &out)
{
	static hls::stream<uint256_dt> streamArray[2];
	static hls::stream<uint512_dt> rd_buffer;
	static hls::stream<uint512_dt> wr_buffer;

	// depth of rd_buffer and wr_buffer set such that burst transfers can be supported.
	#pragma HLS STREAM variable = streamArray depth = 10
	#pragma HLS STREAM variable = rd_buffer depth = max_depth_16
	#pragma HLS STREAM variable = wr_buffer depth = max_depth_16

	int end_index = (xdim0 >> SHIFT_BITS);
	int end_row = ydim0;
	unsigned int total_itr_256 = end_row * end_index * batch;
	unsigned int total_itr_512 = (end_row * end_index * batch + 1) >> 1;

	#pragma HLS DATAFLOW
	read_grid(arg0, rd_buffer, total_itr_512);
	stream_convert_512_256(rd_buffer, streamArray[0], total_itr_512, total_itr_256);
	fifo256_2axis(streamArray[0], out, total_itr_256);
	axis2_fifo256(in, streamArray[1], total_itr_256);
	stream_convert_256_512(streamArray[1], wr_buffer, total_itr_512, total_itr_256);
	write_grid(arg1, wr_buffer, total_itr_512);

}
extern "C" {

	void stencil_mem2stream(
			uint512_dt* arg0,
			uint512_dt* arg1,
			const int count,
			const int xdim0,
			const int ydim0,
			const int batch,
			hls::stream <t_pkt> &in,
			hls::stream <t_pkt> &out)
	{
			#pragma HLS INTERFACE depth=4096 m_axi port = arg0 offset = slave bundle = gmem0 max_read_burst_length=64 max_write_burst_length=64 \
									num_read_outstanding=4 num_write_outstanding=4
			#pragma HLS INTERFACE depth=4096 m_axi port = arg1 offset = slave bundle = gmem1
			#pragma HLS INTERFACE s_axilite port = arg0 bundle = control
			#pragma HLS INTERFACE s_axilite port = arg1 bundle = control
			#pragma HLS INTERFACE s_axilite port = count bundle = control
			#pragma HLS INTERFACE s_axilite port = xdim0 bundle = control
			#pragma HLS INTERFACE s_axilite port = ydim0 bundle = control
			#pragma HLS INTERFACE s_axilite port = batch bundle = control
			#pragma HLS INTERFACE axis port = in  register
			#pragma HLS INTERFACE axis port = out register
			#pragma HLS INTERFACE s_axilite port = return bundle = control


			for (int i = 0; i < count; i++)
			{
				process_mem2stream(arg0, arg1, count, xdim0, ydim0, batch, in, out);
				process_mem2stream(arg1, arg0, count, xdim0, ydim0, batch, in, out);
			}
	}
}

