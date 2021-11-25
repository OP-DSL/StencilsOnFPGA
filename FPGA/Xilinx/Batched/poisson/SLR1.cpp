#include <ap_int.h>
#include <hls_stream.h>
#include <ap_axi_sdata.h>
#include <math.h>

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

    // Unrolling iterative loop
    for(int i = 0; i < SLR1_P_STAGE/2; i++){
		#pragma HLS unroll
		process_grid( streamArray[i], streamArray[i+1], data_g);
	}

	fifo256_2axis(streamArray[SLR1_P_STAGE/2], out1, data_g);
	axis2_fifo256(in2, streamArray[SLR1_P_STAGE/2+1], data_g);

	// unrolling iterative loop
    for(int i = SLR1_P_STAGE/2+1; i < SLR1_P_STAGE+1; i++){
		#pragma HLS unroll
		process_grid( streamArray[i], streamArray[i+1], data_g);
	}

	fifo256_2axis(streamArray[SLR1_P_STAGE+1], out2, data_g);


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
