#include <ap_int.h>
#include <hls_stream.h>
#include <ap_axi_sdata.h>
#include<math.h>

#include "stencil.h"
#include "stencil.cpp"

void process_SLR (hls::stream <t_pkt> &in, hls::stream <t_pkt> &out,
		const int xdim0_poisson_kernel_stencil, const int base0, const int xdim1_poisson_kernel_stencil, const int base1, const int size0, int size1, int batches){


    static hls::stream<uint256_dt> streamArray[40 + 1];
    #pragma HLS STREAM variable = streamArray depth = 2

    struct data_G data_g;
	data_g.end_index = (xdim0_poisson_kernel_stencil >> SHIFT_BITS);
	data_g.end_row = size1+2;
	data_g.outer_loop_limit = size1+3;
	data_g.gridsize = data_g.outer_loop_limit * data_g.end_index * batches;
	data_g.endindex_minus1 = data_g.end_index -1;
	data_g.endrow_plus1 = data_g.end_row + 1;
	data_g.endrow_plus2 = data_g.end_row + 2;
	data_g.endrow_minus1 = data_g.end_row - 1;
	data_g.total_itr_512 = data_g.end_row * (data_g.end_index >> 1) * batches;

	#pragma HLS dataflow
    axis2_fifo256(in, streamArray[0], data_g);

    process_grid( streamArray[0], streamArray[1], size0, size1, xdim0_poisson_kernel_stencil, data_g);
    process_grid( streamArray[1], streamArray[2], size0, size1, xdim0_poisson_kernel_stencil, data_g);
    process_grid( streamArray[2], streamArray[3], size0, size1, xdim0_poisson_kernel_stencil, data_g);
    process_grid( streamArray[3], streamArray[4], size0, size1, xdim0_poisson_kernel_stencil, data_g);

    process_grid( streamArray[4], streamArray[5], size0, size1, xdim0_poisson_kernel_stencil, data_g);
    process_grid( streamArray[5], streamArray[6], size0, size1, xdim0_poisson_kernel_stencil, data_g);
    process_grid( streamArray[6], streamArray[7], size0, size1, xdim0_poisson_kernel_stencil, data_g);
    process_grid( streamArray[7], streamArray[8], size0, size1, xdim0_poisson_kernel_stencil, data_g);

    process_grid( streamArray[8], streamArray[9], size0, size1, xdim0_poisson_kernel_stencil,   data_g);
    process_grid( streamArray[9], streamArray[10], size0, size1, xdim0_poisson_kernel_stencil,  data_g);
    process_grid( streamArray[10], streamArray[11], size0, size1, xdim0_poisson_kernel_stencil, data_g);
    process_grid( streamArray[11], streamArray[12], size0, size1, xdim0_poisson_kernel_stencil, data_g);

    process_grid( streamArray[12], streamArray[13], size0, size1, xdim0_poisson_kernel_stencil, data_g);
    process_grid( streamArray[13], streamArray[14], size0, size1, xdim0_poisson_kernel_stencil, data_g);
    process_grid( streamArray[14], streamArray[15], size0, size1, xdim0_poisson_kernel_stencil, data_g);
    process_grid( streamArray[15], streamArray[16], size0, size1, xdim0_poisson_kernel_stencil, data_g);

    process_grid( streamArray[16], streamArray[17], size0, size1, xdim0_poisson_kernel_stencil, data_g);
    process_grid( streamArray[17], streamArray[18], size0, size1, xdim0_poisson_kernel_stencil, data_g);
    process_grid( streamArray[18], streamArray[19], size0, size1, xdim0_poisson_kernel_stencil, data_g);
    process_grid( streamArray[19], streamArray[20], size0, size1, xdim0_poisson_kernel_stencil, data_g);

	fifo256_2axis(streamArray[20], out, data_g);


}


// top function for kernel in SLR2
// this kernel communicates to/from SLR1
extern "C" {
void stencil_SLR2(
		const int base0,
		const int base1,
		const int size0,
		const int size1,
		const int xdim0_poisson_kernel_stencil,
		const int xdim1_poisson_kernel_stencil,
		const int count,
		const int batches,
		hls::stream <t_pkt> &in,
		hls::stream <t_pkt> &out
		){

	#pragma HLS INTERFACE axis port = in register
	#pragma HLS INTERFACE axis port = out register
	#pragma HLS INTERFACE s_axilite port = base0 bundle = control
	#pragma HLS INTERFACE s_axilite port = base1 bundle = control
	#pragma HLS INTERFACE s_axilite port = size0 bundle = control
	#pragma HLS INTERFACE s_axilite port = size1 bundle = control
	#pragma HLS INTERFACE s_axilite port = xdim0_poisson_kernel_stencil bundle = control
	#pragma HLS INTERFACE s_axilite port = xdim1_poisson_kernel_stencil bundle = control
	#pragma HLS INTERFACE s_axilite port = count bundle = control
	#pragma HLS INTERFACE s_axilite port = batches bundle = control
	#pragma HLS INTERFACE s_axilite port = return bundle = control


	// iterative loop
	for(int i =  0; i < 2*count ; i++){
		#pragma HLS dataflow
		process_SLR( in, out, xdim0_poisson_kernel_stencil, base0, xdim1_poisson_kernel_stencil, base1, size0, size1, batches);
	}

}
}
