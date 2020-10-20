#include <ap_int.h>
#include <hls_stream.h>
#include <ap_axi_sdata.h>
#include <math.h>
#include "stencil.h"
#include "stencil.cpp"


// coalesced memory access at 512 bit to get maximum out of memory bandwidth
// Single pipelined loop below will be mapped to single memory transfer 
// which will further split into multiple transfers by axim module.
static void read_grid(uint512_dt*  arg0, hls::stream<uint512_dt> &rd_buffer, struct data_G data_g){
	unsigned int total_itr = data_g.total_itr_512;
	for (int itr = 0; itr < end_row; itr++){
		#pragma HLS PIPELINE II=1
		#pragma HLS loop_tripcount min=min_grid max=max_grid avg=avg_grid
		rd_buffer << arg0[itr];
	}
}

// data width conversion to support 256 bit width compute pipeline
static void stream_convert_512_256(hls::stream<uint512_dt> &in, hls::stream<uint256_dt> &out, struct data_G data_g){
	unsigned int total_itr = data_g.total_ter_512;
	for (int itr = 0; itr < total_itr; itr++){
		#pragma HLS PIPELINE II=2
		#pragma HLS loop_tripcount min=min_grid max=max_grid avg=avg_grid
		uint512_dt tmp = in.read();
		uint256_dt var_l = tmp.range(255,0);
		uint256_dt var_h = tmp.range(511,256);;
		out << var_l;
		out << var_h;
	}
}

// data width conversion to support 512 bit width memory write interface
static void stream_convert_256_512(hls::stream<uint256_dt> &in, hls::stream<uint512_dt> &out, struct data_G data_g){
	unsigned int total_itr = data_g.total_ter_512;
	for (int itr = 0; itr < total_itr; itr++){
		#pragma HLS PIPELINE II=2
		#pragma HLS loop_tripcount min=min_grid max=max_grid avg=avg_grid
		uint512_dt tmp;
		tmp.range(255,0) = in.read();
		tmp.range(511,256) = in.read();
		out << tmp;
	}
}


// task level parallelism is applied using HLS data flow directive
// stream interface is used to connect the whole pipeline
void process_SLR0 (uint512_dt*  arg0, uint512_dt*  arg1, hls::stream <t_pkt> &in, hls::stream <t_pkt> &out,
		const int xdim0_poisson_kernel_stencil, const int base0, const int xdim1_poisson_kernel_stencil,
		const int base1, const int size0, int size1, const int batches){


    static hls::stream<uint256_dt> streamArray[40 + 1];
    static hls::stream<uint512_dt> rd_buffer;
    static hls::stream<uint512_dt> wr_buffer;

	// depth of rd_buffer and wr_buffer set such that burst transfers can be supported. 
    #pragma HLS STREAM variable = streamArray depth = 2
	#pragma HLS STREAM variable = rd_buffer depth = max_depth_16
	#pragma HLS STREAM variable = wr_buffer depth = max_depth_16

    struct data_G data_g;
	data_g.end_index = (xdim0_poisson_kernel_stencil >> SHIFT_BITS); // number of blocks with V number of elements to be processed in a single row
	data_g.end_row = size1+2; // includes the boundary
	data_g.outer_loop_limit = size1+3; // n + D/2
	data_g.gridsize = data_g.outer_loop_limit * data_g.end_index * batches;
	data_g.endindex_minus1 = data_g.end_index -1;
	data_g.endrow_plus1 = data_g.end_row + 1;
	data_g.endrow_plus2 = data_g.end_row + 2;
	data_g.endrow_minus1 = data_g.end_row - 1;
	data_g.total_itr_512 = data_g.end_row * (data_g.end_index >> 1) * batches;


	// parallel execution of following modules
	#pragma HLS dataflow
	read_grid(arg0, rd_buffer, data_g);
	stream_convert_512_256(rd_buffer, streamArray[0], data_g);

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

	// seding data to kernel which resides in another SLR
	fifo256_2axis(streamArray[20], out, xdim0_poisson_kernel_stencil, base0, size1, batches);
	// getting data from kernel which resides in another SLR
	axis2_fifo256(in, streamArray[21], xdim0_poisson_kernel_stencil, base0, size1, batches);

	stream_convert_256_512(streamArray[21], wr_buffer, data_g);
	write_grid(arg1, wr_buffer, data_g);

}


//-DHOST_CODE_OPT -DLOCAL_BUF_OPT -DDF_OPT -DFP_OPT
//--sc stencil_SLR0_1.out:stencil_SLR1_1.in
//--sc stencil_SLR1_1.out:stencil_SLR0_1.in


// top level function for SLR0, includes modules for transferring data from external memory and compute pipeline
// arg0 and arg1 will be mapped into a AXI port
// additionally there are two stream interfaces to support kernel2kernel communication
// iterative loop is moved inside this top level function to reduce kernel call overhead

extern "C" {
void stencil_SLR0(
		uint512_dt*  arg0,
		uint512_dt*  arg1,
		const int base0,
		const int base1,
		const int size0,
		const int size1,
		const int xdim0_poisson_kernel_stencil,
		const int xdim1_poisson_kernel_stencil,
		const int count,
		const int batches,
		hls::stream <t_pkt> &in,
		hls::stream <t_pkt> &out){

	#pragma HLS INTERFACE depth=4096 m_axi port = arg0 offset = slave bundle = gmem0 max_read_burst_length=64 max_write_burst_length=64 \
							num_read_outstanding=4 num_write_outstanding=4
	#pragma HLS INTERFACE depth=4096 m_axi port = arg1 offset = slave bundle = gmem0
	#pragma HLS INTERFACE s_axilite port = arg0 bundle = control
	#pragma HLS INTERFACE s_axilite port = arg1 bundle = control
	#pragma HLS INTERFACE axis port = in  register
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
	for(int i =  0; i < count; i++){
		process_SLR0(arg0, arg1, in, out, xdim0_poisson_kernel_stencil, base0, xdim0_poisson_kernel_stencil, base0, size0, size1, batches);
		process_SLR0(arg1, arg0, in, out, xdim0_poisson_kernel_stencil, base0, xdim0_poisson_kernel_stencil, base0, size0, size1, batches);
	}

}
}
