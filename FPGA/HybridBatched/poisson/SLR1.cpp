#include <ap_int.h>
#include <hls_stream.h>
#include <ap_axi_sdata.h>
#include <math.h>
#include <stdio.h>
#include "stencil.h"
#include "stencil.cpp"


void process_SLR (hls::stream <t_pkt> &in1, hls::stream <t_pkt> &out1, hls::stream <t_pkt> &in2, hls::stream <t_pkt> &out2,
		const int xdim0, const int size0, int size1, int batches){


    static hls::stream<uint256_dt> streamArray[40 + 1];
    #pragma HLS STREAM variable = streamArray depth = 2

    struct data_G data_g;
    data_g.sizex = size0;
    data_g.sizey = size1;
    data_g.xdim0 = xdim0;
	data_g.end_index = (xdim0 >> SHIFT_BITS);
	data_g.end_row = size1+2;
	data_g.outer_loop_limit = size1+3;
	data_g.gridsize = (data_g.end_row* batches + 1) * data_g.end_index;
	data_g.endindex_minus1 = data_g.end_index -1;
	data_g.endrow_plus1 = data_g.end_row + 1;
	data_g.endrow_plus2 = data_g.end_row + 2;
	data_g.endrow_minus1 = data_g.end_row - 1;
	data_g.total_itr_256 = data_g.end_row * data_g.end_index * batches;
	data_g.total_itr_512 = (data_g.end_row * data_g.end_index * batches + 1) >> 1;

	// pipeline is broken into two stages to help SLR crossing through
	// higher register depth for both of two AXI stream interfaces
	#pragma HLS dataflow
    axis2_fifo256(in1, streamArray[0], data_g);

    process_grid( streamArray[0], streamArray[1], data_g);
    process_grid( streamArray[1], streamArray[2], data_g);
    process_grid( streamArray[2], streamArray[3], data_g);
    process_grid( streamArray[3], streamArray[4], data_g);

    process_grid( streamArray[4], streamArray[5], data_g);
    process_grid( streamArray[5], streamArray[6], data_g);
    process_grid( streamArray[6], streamArray[7], data_g);
    process_grid( streamArray[7], streamArray[8], data_g);

    process_grid( streamArray[8], streamArray[9], data_g);
    process_grid( streamArray[9], streamArray[10], data_g);


	fifo256_2axis(streamArray[10], out1, data_g);
	axis2_fifo256(in2, streamArray[20], data_g);

	process_grid( streamArray[20], streamArray[21], data_g);
	process_grid( streamArray[21], streamArray[22], data_g);
	process_grid( streamArray[22], streamArray[23], data_g);
	process_grid( streamArray[23], streamArray[24], data_g);

	process_grid( streamArray[24], streamArray[25], data_g);
	process_grid( streamArray[25], streamArray[26], data_g);
	process_grid( streamArray[26], streamArray[27], data_g);
	process_grid( streamArray[27], streamArray[28], data_g);

	process_grid( streamArray[28], streamArray[29], data_g);
	process_grid( streamArray[29], streamArray[30], data_g);


	fifo256_2axis(streamArray[30], out2, data_g);


}

// top level function for SLR2 
// this kernel doesn't interact with external memory
// but it gets data to/from SLRO and SLR1 kernels

extern "C" {
void stencil_SLR1(
		const int size0,
		const int size1,
		const int xdim0,
		const int count,
		const int batches,
		hls::stream <t_pkt> &in1,
		hls::stream <t_pkt> &out1,
		hls::stream <t_pkt> &in2,
		hls::stream <t_pkt> &out2
		){

	#pragma HLS INTERFACE axis port = in1 register
	#pragma HLS INTERFACE axis port = out1 register
	#pragma HLS INTERFACE axis port = in2 register
	#pragma HLS INTERFACE axis port = out2 register
	#pragma HLS INTERFACE s_axilite port = size0 bundle = control
	#pragma HLS INTERFACE s_axilite port = size1 bundle = control
	#pragma HLS INTERFACE s_axilite port = xdim0 bundle = control
	#pragma HLS INTERFACE s_axilite port = count bundle = control
	#pragma HLS INTERFACE s_axilite port = batches bundle = control
	#pragma HLS INTERFACE s_axilite port = return bundle = control


	// iterative loop
	for(int i =  0; i < 2*count ; i++){
		#pragma HLS dataflow
		process_SLR( in1, out1,in2, out2, xdim0, size0, size1, batches);
	}

}
}
