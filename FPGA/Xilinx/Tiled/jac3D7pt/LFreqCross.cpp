#include <ap_int.h>
#include <hls_stream.h>
#include <ap_axi_sdata.h>

typedef ap_uint<512> uint512_dt;
typedef ap_uint<256> uint256_dt;
typedef ap_axiu<512,0,0,0> t_pkt;

#define MAX_SIZE_X 2048
#define MAX_DEPTH_16 (MAX_SIZE_X/16)

//user function
#define PORT_WIDTH 8
#define SHIFT_BITS 3
#define DATATYPE_SIZE 32
//#define BEAT_SHIFT_BITS 10
#define BURST_LEN MAX_DEPTH_16

#define STAGES 2

const int max_size_y = MAX_SIZE_X;
const int min_size_y = 20;
const int avg_size_y = MAX_SIZE_X;

const int max_block_x = MAX_SIZE_X/16 + 1;
const int min_block_x = 20/16 + 1;
const int avg_block_x = MAX_SIZE_X/16 + 1;

const int max_grid = max_block_x * max_size_y;
const int min_grid = min_block_x * min_size_y;
const int avg_grid = avg_block_x * avg_size_y;

const int port_width  = PORT_WIDTH;
const int max_depth_16 = MAX_DEPTH_16;
const int max_depth_8 = MAX_DEPTH_16*2;


void process_LFreqCross (hls::stream <t_pkt> &in, hls::stream <t_pkt> &out, const int xdim0_poisson_kernel_stencil,
		const int base0, const int xdim1_poisson_kernel_stencil, const int base1, const int size0, int size1){
	#pragma HLS dataflow
	int end_index = (xdim1_poisson_kernel_stencil >> (SHIFT_BITS+1));
	int End_row = size1+3;
	for (int itr = 0; itr < End_row * (end_index); itr++){
		#pragma HLS loop_tripcount min=min_grid max=max_grid avg=avg_grid
		#pragma HLS PIPELINE II=1
		out.write(in.read());
	}
}


//-DHOST_CODE_OPT -DLOCAL_BUF_OPT -DDF_OPT -DFP_OPT
//--sc stencil_SLR0_1.out:stencil_SLR1_1.in
//--sc stencil_SLR1_1.out:stencil_SLR0_1.in


//stream_connect=stencil_SLR0_1.out:LFreqCross_1.in
//stream_connect=LFreqCross_1.out:LFreqCross_2.in
//
//
//stream_connect=LFreqCross_2.out:stencil_SLR1_1.in1
//stream_connect=stencil_SLR1_1.out1:LFreqCross_3.in
//stream_connect=LFreqCross_3.out:LFreqCross_4.in
//
//stream_connect=LFreqCross_4.out:stencil_SLR2_1.in
//stream_connect=stencil_SLR2_1.out:LFreqCross_5.in
//stream_connect=LFreqCross_5.out:LFreqCross_6.in
//
//stream_connect=LFreqCross_6.out:stencil_SLR1_1.in2
//stream_connect=stencil_SLR1_1.out2:LFreqCross_7.in
//stream_connect=LFreqCross_7.out:LFreqCross_8.in
//
//stream_connect=LFreqCross_8.out:stencil_SLR0_1.in

extern "C" {
void LFreqCross(
		const int base0,
		const int base1,
		const int size0,
		const int size1,
		const int xdim0_poisson_kernel_stencil,
		const int xdim1_poisson_kernel_stencil,
		const int count,
		hls::stream <t_pkt> &in,
		hls::stream <t_pkt> &out){

	#pragma HLS INTERFACE axis port = in register
	#pragma HLS INTERFACE axis port = out register
	#pragma HLS INTERFACE s_axilite port = base0 bundle = control
	#pragma HLS INTERFACE s_axilite port = base1 bundle = control
	#pragma HLS INTERFACE s_axilite port = size0 bundle = control
	#pragma HLS INTERFACE s_axilite port = size1 bundle = control
	#pragma HLS INTERFACE s_axilite port = xdim0_poisson_kernel_stencil bundle = control
	#pragma HLS INTERFACE s_axilite port = xdim1_poisson_kernel_stencil bundle = control
	#pragma HLS INTERFACE s_axilite port = count bundle = control
	#pragma HLS INTERFACE s_axilite port = return bundle = control


	for(int i =  0; i < 2*count; i++){
		process_LFreqCross(in, out, xdim0_poisson_kernel_stencil, base0, xdim0_poisson_kernel_stencil, base0, size0, size1);
	}

}
}
