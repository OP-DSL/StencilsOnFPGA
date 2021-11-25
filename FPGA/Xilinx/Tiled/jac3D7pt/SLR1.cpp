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




void process_SLR (hls::stream <t_pkt_1024> &in1l, hls::stream <t_pkt_1024> &in1u, hls::stream <t_pkt_1024> &out1l, hls::stream <t_pkt_1024> &out1u,
		hls::stream <t_pkt_1024> &in2l, hls::stream <t_pkt_1024> &in2u, hls::stream <t_pkt_1024> &out2l, hls::stream <t_pkt_1024> &out2u,
		const int xdim0, const unsigned short offset_x, const unsigned short tile_x,
		unsigned short offset_y, unsigned short tile_y,
		const unsigned short size_x, const unsigned short size_y, const unsigned short size_z){

    static hls::stream<uint256_dt> streamC_4_256_0[8];
    static hls::stream<uint256_dt> streamC_4_256_1[8];
    static hls::stream<uint256_dt> streamC_4_256_2[8];
    static hls::stream<uint256_dt> streamC_4_256_3[8];

    static hls::stream<uint288_dt> streamC_4_288_0[8];
    static hls::stream<uint288_dt> streamC_4_288_1[8];
    static hls::stream<uint288_dt> streamC_4_288_2[8];
    static hls::stream<uint288_dt> streamC_4_288_3[8];

	#pragma HLS STREAM variable = streamC_4_256_0 depth = 2
	#pragma HLS STREAM variable = streamC_4_256_1 depth = 2
	#pragma HLS STREAM variable = streamC_4_256_2 depth = 2
	#pragma HLS STREAM variable = streamC_4_256_3 depth = 2

	#pragma HLS STREAM variable = streamC_4_288_0 depth = 2
	#pragma HLS STREAM variable = streamC_4_288_1 depth = 2
	#pragma HLS STREAM variable = streamC_4_288_2 depth = 2
	#pragma HLS STREAM variable = streamC_4_288_3 depth = 2

    struct data_G data_g;
    data_g.sizex = size_x;
    data_g.sizey = size_y;
    data_g.sizez = size_z;

    data_g.offset_x = offset_x;
    data_g.tile_x = tile_x;
    data_g.offset_y = offset_y;
    data_g.tile_y = tile_y;


	data_g.xblocks = (tile_x >> (SHIFT_BITS+3));
	data_g.grid_sizey = size_y + 2;
	data_g.grid_sizez = size_z+2;
	data_g.limit_z = size_z+3;

	unsigned short tiley_1 = (data_g.tile_y - 1);
	unsigned int plane_size = data_g.xblocks * data_g.tile_y;

	data_g.plane_diff = data_g.xblocks * tiley_1;
	data_g.line_diff = data_g.xblocks - 1;
	data_g.gridsize_pr = plane_size * (data_g.limit_z);

	unsigned int gridsize_da = plane_size * (data_g.grid_sizez);



	#pragma HLS dataflow

//	axis2_fifo256_8(in1l, in1u,streamC_4_256_0[0], streamC_4_256_0[1],
//				streamC_4_256_1[0], streamC_4_256_1[1],
//				streamC_4_256_2[0], streamC_4_256_2[1],
//				streamC_4_256_3[0], streamC_4_256_3[1], gridsize_da);


    axis2_fifo288_8(in1l, in1u,streamC_4_288_0[0], streamC_4_288_0[1],
			streamC_4_288_1[0], streamC_4_288_1[1],
			streamC_4_288_2[0], streamC_4_288_2[1],
			streamC_4_288_3[0], streamC_4_288_3[1], gridsize_da);


    process_tile( streamC_4_288_0[0], streamC_4_288_0[1],
			streamC_4_256_0[0], streamC_4_256_0[1],
			-1, data_g);

    process_tile( streamC_4_288_1[0], streamC_4_288_1[1],
			streamC_4_256_1[0], streamC_4_256_1[1],
			15, data_g);

    process_tile( streamC_4_288_2[0], streamC_4_288_2[1],
			streamC_4_256_2[0], streamC_4_256_2[1],
			31, data_g);

    process_tile( streamC_4_288_3[0], streamC_4_288_3[1],
			streamC_4_256_3[0], streamC_4_256_3[1],
			47, data_g);



	fifo256_8_2axis1(streamC_4_256_0[0], streamC_4_256_0[1],
			streamC_4_256_1[0], streamC_4_256_1[1],
			streamC_4_256_2[0], streamC_4_256_2[1],
			streamC_4_256_3[0], streamC_4_256_3[1], out1l, out1u, gridsize_da);

	axis2_fifo256_8(in2l, in2u,streamC_4_256_0[4], streamC_4_256_0[5],
			streamC_4_256_1[4], streamC_4_256_1[5],
			streamC_4_256_2[4], streamC_4_256_2[5],
			streamC_4_256_3[4], streamC_4_256_3[5], gridsize_da);

	fifo256_8_2axis(streamC_4_256_0[4], streamC_4_256_0[5],
			streamC_4_256_1[4], streamC_4_256_1[5],
			streamC_4_256_2[4], streamC_4_256_2[5],
			streamC_4_256_3[4], streamC_4_256_3[5], out2l, out2u, gridsize_da);
}

extern "C" {
void stencil_SLR1(
		const int tilex_count,
		const int tiley_count,
		const int sizex,
		const int sizey,
		const int sizez,
		const int xdim0,
		const int count,

		hls::stream <t_pkt_32> &tile_s_in,
		hls::stream <t_pkt_32> &tile_s_out,

		hls::stream <t_pkt_1024> &in1l,
		hls::stream <t_pkt_1024> &in1u,
		hls::stream <t_pkt_1024> &out1l,
		hls::stream <t_pkt_1024> &out1u,
		hls::stream <t_pkt_1024> &in2l,
		hls::stream <t_pkt_1024> &in2u,
		hls::stream <t_pkt_1024> &out2l,
		hls::stream <t_pkt_1024> &out2u
		){

	#pragma HLS INTERFACE axis port = in1l register
	#pragma HLS INTERFACE axis port = in1u register
	#pragma HLS INTERFACE axis port = out1l register
	#pragma HLS INTERFACE axis port = out1u register
	#pragma HLS INTERFACE axis port = in2l register
	#pragma HLS INTERFACE axis port = in2u register
	#pragma HLS INTERFACE axis port = out2l register
	#pragma HLS INTERFACE axis port = out2u register
	#pragma HLS INTERFACE axis port = tile_s_in register
	#pragma HLS INTERFACE axis port = tile_s_out register

	#pragma HLS INTERFACE s_axilite port = tilex_count bundle = control
	#pragma HLS INTERFACE s_axilite port = tiley_count bundle = control
	#pragma HLS INTERFACE s_axilite port = sizey bundle = control
	#pragma HLS INTERFACE s_axilite port = sizex bundle = control
	#pragma HLS INTERFACE s_axilite port = sizez bundle = control
	#pragma HLS INTERFACE s_axilite port = xdim0 bundle = control
	#pragma HLS INTERFACE s_axilite port = count bundle = control
	#pragma HLS INTERFACE s_axilite port = return bundle = control

	unsigned int tile_memx[256];
	unsigned int tile_memy[256];

	for(int j = 0; j < tilex_count; j++){
		t_pkt_32 tmp_s;
		tmp_s = tile_s_in.read();
		tile_s_out.write(tmp_s);
		tile_memx[j] = tmp_s.data;
	}

	for(int j = 0; j < tiley_count; j++){
		t_pkt_32 tmp_s;
		tmp_s = tile_s_in.read();
		tile_s_out.write(tmp_s);
		tile_memy[j] = tmp_s.data;
	}

	unsigned int total_count = (count << 1) * tilex_count*tiley_count;
	unsigned short j = 0, k = 0;
	unsigned short j_dum = 0, k_dum = 0;
	unsigned char tilex_count_s = tilex_count;
	unsigned char tiley_count_s = tiley_count;
	unsigned char tile_count = tilex_count_s * tiley_count_s;
	for(unsigned int itr= 0;  itr < total_count; itr++){
		bool cond_k = (k == tilex_count - 1);
		bool cond_j = (j == tiley_count - 1);

		if(cond_k){
			k_dum = 0;
		} else {
			k_dum = k + 1;
		}

		if(cond_j && cond_k){
			j_dum = 0;
		} else if(cond_k){
			j_dum = j + 1;
		}

		unsigned short offset_x = tile_memx[k] & 0xffff;
		unsigned short tile_x   = tile_memx[k] >> 16;
		unsigned short offset_y = tile_memy[j] & 0xffff;
		unsigned short tile_y   = tile_memy[j] >> 16;

		j = j_dum;
		k = k_dum;
		process_SLR( in1l, in1u, out1l, out1u, in2l, in2u, out2l, out2u, xdim0, offset_x, tile_x, offset_y, tile_y, sizex, sizey, sizez);
	}

}
}
