#include <ap_int.h>
#include <hls_stream.h>
#include <ap_axi_sdata.h>
#include<math.h>

#include "stencil.h"
#include "stencil.cpp"

void process_SLR (hls::stream <t_pkt> &in, hls::stream <t_pkt> &out,
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

	#pragma HLS dataflow
    axis2_fifo256(in, streamArray[0], data_g);

    // unrolling iterative loop
    for(int i = 0; i < SLR2_P_STAGE; i++){
		#pragma HLS unroll
		process_grid( streamArray[i], streamArray[i+1], data_g);
	}

	fifo256_2axis(streamArray[20], out, data_g);


}


// top function for kernel in SLR2
// this kernel communicates to/from SLR1
extern "C" {
void stencil_SLR2(
		const int size0,
		const int size1,
		const int xdim0,
		const int count,
		const int batches,
		hls::stream <t_pkt> &in,
		hls::stream <t_pkt> &out
		){

	#pragma HLS INTERFACE axis port = in register
	#pragma HLS INTERFACE axis port = out register
	#pragma HLS INTERFACE s_axilite port = size0 bundle = control
	#pragma HLS INTERFACE s_axilite port = size1 bundle = control
	#pragma HLS INTERFACE s_axilite port = xdim0 bundle = control
	#pragma HLS INTERFACE s_axilite port = count bundle = control
	#pragma HLS INTERFACE s_axilite port = batches bundle = control
	#pragma HLS INTERFACE s_axilite port = return bundle = control


	// iterative loop
	for(int i =  0; i < 2*count ; i++){
		#pragma HLS dataflow
		process_SLR( in, out, xdim0, size0, size1, batches);
	}

}
}
