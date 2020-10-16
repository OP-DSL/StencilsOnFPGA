#include <ap_int.h>
#include <hls_stream.h>
#include <ap_axi_sdata.h>
#include <math.h>
#include <stencil.h>
#include <stencil.cpp>


//-DHOST_CODE_OPT -DLOCAL_BUF_OPT -DDF_OPT -DFP_OPT
//--sc ops_poisson_kernel_stencil_SLR0.out:ops_poisson_kernel_stencil_SLR1.in
//--sc ops_poisson_kernel_stencil_SLR0.in:ops_poisson_kernel_stencil_SLR1.out


//--------------------------------------------------------------------------------------------------------------------------
//-------------------------------------------------------- SLR crossing SLR1 -----------------------------------------------
//--------------------------------------------------------------------------------------------------------------------------




void process_SLR (hls::stream <t_pkt> &in1, hls::stream <t_pkt> &out1, hls::stream <t_pkt> &in2, hls::stream <t_pkt> &out2,
		const int xdim0, const unsigned short size_x, const unsigned short size_y, const unsigned short size_z, const unsigned short batches){

    static hls::stream<uint256_dt> streamArray[40 + 1];
    #pragma HLS STREAM variable = streamArray depth = 2

    struct data_G data_g;
    data_g.sizex = size_x;
    data_g.sizey = size_y;
    data_g.sizez = size_z;

    data_g.offset_x = 0;
    data_g.tile_x = xdim0;
    data_g.offset_y = 0;
    data_g.tile_y = size_y+2;

    data_g.batches = batches;


	data_g.xblocks = (data_g.tile_x >> SHIFT_BITS);
	data_g.grid_sizey = size_y + 2;
	data_g.grid_sizez = size_z+2;
	data_g.limit_z = size_z+3;

	unsigned short tiley_1 = (data_g.tile_y - 1);
	unsigned int plane_size = data_g.xblocks * data_g.tile_y;

	data_g.plane_diff = data_g.xblocks * tiley_1;
	data_g.line_diff = data_g.xblocks - 1;
	data_g.gridsize_pr = plane_size * register_it<unsigned int>(data_g.grid_sizez * batches+1);

	unsigned int gridsize_da = register_it<unsigned int>(plane_size * (data_g.grid_sizez)) * batches;
	data_g.gridsize_da = gridsize_da;



	#pragma HLS dataflow
    axis2_fifo256(in1, streamArray[0], gridsize_da);

    process_tile( streamArray[0], streamArray[1], data_g);
    process_tile( streamArray[1], streamArray[2], data_g);
    process_tile( streamArray[2], streamArray[3], data_g);
    process_tile( streamArray[3], streamArray[4], data_g);
    process_tile( streamArray[4], streamArray[5], data_g);



	fifo256_2axis(streamArray[5], out1, gridsize_da);
	axis2_fifo256(in2, streamArray[40], gridsize_da);

	process_tile( streamArray[40], streamArray[6], data_g);
	process_tile( streamArray[6], streamArray[7], data_g);
	process_tile( streamArray[7], streamArray[8], data_g);
	process_tile( streamArray[8], streamArray[9], data_g);
	process_tile( streamArray[9], streamArray[10], data_g);
//	process_tile( streamArray[10], streamArray[11], data_g);

	fifo256_2axis(streamArray[10], out2, gridsize_da);


}

extern "C" {
void stencil_SLR1(
		const int sizex,
		const int sizey,
		const int sizez,
		const int xdim0,
		const int batches,
		const int count,


		hls::stream <t_pkt> &in1,
		hls::stream <t_pkt> &out1,
		hls::stream <t_pkt> &in2,
		hls::stream <t_pkt> &out2
		){

	#pragma HLS INTERFACE axis port = in1 register
	#pragma HLS INTERFACE axis port = out1 register
	#pragma HLS INTERFACE axis port = in2 register
	#pragma HLS INTERFACE axis port = out2 register

	#pragma HLS INTERFACE s_axilite port = sizey bundle = control
	#pragma HLS INTERFACE s_axilite port = sizex bundle = control
	#pragma HLS INTERFACE s_axilite port = sizez bundle = control
	#pragma HLS INTERFACE s_axilite port = xdim0 bundle = control
	#pragma HLS INTERFACE s_axilite port = batches bundle = control
	#pragma HLS INTERFACE s_axilite port = count bundle = control
	#pragma HLS INTERFACE s_axilite port = return bundle = control


	for(unsigned short itr =  0; itr < 2*count ; itr++){
			process_SLR( in1, out1, in2, out2, xdim0, sizex, sizey, sizez, batches);

	}

}
}
