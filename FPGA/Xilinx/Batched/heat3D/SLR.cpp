#include <ap_int.h>
#include <hls_stream.h>
#include <ap_axi_sdata.h>
#include <math.h>
#include "stencil.h"
#include "stencil.cpp"

void process_SLR(hls::stream <t_pkt> &in, hls::stream<t_pkt> &out, const int xdim0, const unsigned short size_x,
		const unsigned int size_y, const unsigned int size_z, const unsigned short batches, const float calcParam_K)
{
	hls::stream<uint256_dt> streamArray[SLR_P_STAGE + 1];
#pragma HLS STREAM variable = streamArray depth = 10

	data_G data_g;
	data_g.sizex = size_x;
	data_g.sizey = size_y;
	data_g.sizez = size_z;
	data_g.offset_x = 0;
	data_g.grid_size_x = xdim0;
	data_g.xblocks = (data_g.grid_size_x >> SHIFT_BITS);
	data_g.offset_y = 0;
	data_g.grid_size_y = size_y + 2;
	data_g.offset_z = 0;
	data_g.grid_size_z = size_z + 2;
	data_g.batches = batches;
	data_g.limit_z = size_z + 3;

	unsigned short tile_y_1 = data_g.grid_size_y - 1;
	unsigned int plane_size = data_g.xblocks * data_g.grid_size_y;

	data_g.plane_diff = data_g.xblocks * tile_y_1;
	data_g.line_diff = data_g.xblocks - 1;
	data_g.gridsize_pr = plane_size * register_it(data_g.grid_size_z * batches + 1);
	data_g.gridsize_da = register_it(plane_size * data_g.grid_size_z) * batches;

	const float coefficients[7] = {calcParam_K, calcParam_K, calcParam_K, 1-6*calcParam_K, calcParam_K, calcParam_K, calcParam_K};
#pragma HLS ARRAY_PARTITION variable=coefficients complete dim=1

#pragma HLS DATAFLOW
	{
		axis2_fifo256(in, streamArray[0], data_g.gridsize_da);

		for (int i = 0; i < SLR_P_STAGE; i++)
		{
#pragma HLS unroll
			process_grid(streamArray[i], streamArray[i+1], data_g, coefficients);
		}

		fifo256_2axis(streamArray[SLR_P_STAGE], out, data_g.gridsize_da);

	}

}

extern "C"
{
	void stencil_SLR(
			const int sizex,
			const int sizey,
			const int sizez,
			const int xdim0,
			const int batches,
			const int count,
			const float calcParam_K,
			hls::stream <t_pkt> &in,
			hls::stream <t_pkt> &out)
	{
#pragma HLS INTERFACE axis port = in register
#pragma HLS INTERFACE axis port = out register

#pragma HLS INTERFACE s_axilite port = sizex bundle = control
#pragma HLS INTERFACE s_axilite port = sizey bundle = control
#pragma HLS INTERFACE s_axilite port = sizez bundle = control
#pragma HLS INTERFACE s_axilite port = xdim0 bundle = control
#pragma HLS INTERFACE s_axilite port = batches bundle = control
#pragma HLS INTERFACE s_axilite port = count bundle = control
#pragma HLS INTERFACE s_axilite port = calcParam_K bundle = control
#pragma HLS INTERFACE s_axilite port = return bundle = control

		for (unsigned int i = 0; i < count * 2; i++)
		{
			process_SLR(in, out, xdim0, sizex, sizey, sizez, batches, calcParam_K);
		}

	}
}
