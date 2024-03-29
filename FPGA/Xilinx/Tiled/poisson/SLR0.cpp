#include <ap_int.h>
#include <hls_stream.h>
#include <ap_axi_sdata.h>
#include <math.h>
#include <stencil.h>
#include <stencil.cpp>


static void process_SLR (hls::stream <t_pkt> &in, hls::stream <t_pkt> &out,
				   const int xdim0, const unsigned short offset, const unsigned short tile_x, const unsigned short size_x, const unsigned short size_y){


    static hls::stream<uint256_dt> streamArray[40 + 1];
    static hls::stream<uint512_dt> rd_bufferArr[4];
    static hls::stream<uint512_dt> wr_bufferArr[4];
    static hls::stream<uint512_dt> streamArray_512[4];

    #pragma HLS STREAM variable = streamArray depth = 2
	#pragma HLS STREAM variable = streamArray_512 depth = 2
	#pragma HLS STREAM variable = rd_bufferArr depth = max_depth_16
	#pragma HLS STREAM variable = wr_bufferArr depth = max_depth_16

    struct data_G data_g;
	data_g.end_index = (tile_x >> SHIFT_BITS);
	data_g.end_row = size_y+2;
	data_g.outer_loop_limit = size_y+3;
	data_g.gridsize = (data_g.outer_loop_limit * data_g.end_index);
	data_g.endindex_minus1 = data_g.end_index -1;
	data_g.endrow_plus1 = data_g.end_row + 1;
	data_g.endrow_plus2 = data_g.end_row + 2;
	data_g.endrow_minus1 = data_g.end_row - 1;



	#pragma HLS dataflow
    axis2_fifo256(in, streamArray[0], tile_x, size_y);

    process_tile( streamArray[0], streamArray[1], size_x, size_y, offset, data_g);
    process_tile( streamArray[1], streamArray[2], size_x, size_y, offset, data_g);
    process_tile( streamArray[2], streamArray[3], size_x, size_y, offset, data_g);
    process_tile( streamArray[3], streamArray[4], size_x, size_y, offset, data_g);
    process_tile( streamArray[4], streamArray[5], size_x, size_y, offset, data_g);

	process_tile( streamArray[5], streamArray[6], size_x, size_y, offset, data_g);
	process_tile( streamArray[6], streamArray[7], size_x, size_y, offset, data_g);
	process_tile( streamArray[7], streamArray[8], size_x, size_y, offset, data_g);
	process_tile( streamArray[8], streamArray[9], size_x, size_y, offset, data_g);
	process_tile( streamArray[9], streamArray[10], size_x, size_y, offset, data_g);

    process_tile( streamArray[10], streamArray[11], size_x, size_y, offset, data_g);
    process_tile( streamArray[11], streamArray[12], size_x, size_y, offset, data_g);
    process_tile( streamArray[12], streamArray[13], size_x, size_y, offset, data_g);
    process_tile( streamArray[13], streamArray[14], size_x, size_y, offset, data_g);
    process_tile( streamArray[14], streamArray[15], size_x, size_y, offset, data_g);

	process_tile( streamArray[15], streamArray[16], size_x, size_y, offset, data_g);
	process_tile( streamArray[16], streamArray[17], size_x, size_y, offset, data_g);
	process_tile( streamArray[17], streamArray[18], size_x, size_y, offset, data_g);
	process_tile( streamArray[18], streamArray[19], size_x, size_y, offset, data_g);
	process_tile( streamArray[19], streamArray[20], size_x, size_y, offset, data_g);

	fifo256_2axis(streamArray[20], out, tile_x, size_y);

}


//-DHOST_CODE_OPT -DLOCAL_BUF_OPT -DDF_OPT -DFP_OPT

extern "C" {
void stencil_SLR0(
		const int tile_count,
		const int sizex,
		const int sizey,
		const int xdim0,
		const int count,

		hls::stream <t_pkt_32> &tile_s_in,
		hls::stream <t_pkt_32> &tile_s_out,
		hls::stream <t_pkt> &in,
		hls::stream <t_pkt> &out){

	#pragma HLS INTERFACE axis port = tile_s_out  register
	#pragma HLS INTERFACE axis port = tile_s_in  register
	#pragma HLS INTERFACE axis port = in  register
	#pragma HLS INTERFACE axis port = out register

	#pragma HLS INTERFACE s_axilite port = tile_count bundle = control
	#pragma HLS INTERFACE s_axilite port = sizex bundle = control
	#pragma HLS INTERFACE s_axilite port = sizey bundle = control
	#pragma HLS INTERFACE s_axilite port = xdim0 bundle = control
	#pragma HLS INTERFACE s_axilite port = count bundle = control
	#pragma HLS INTERFACE s_axilite port = return bundle = control




	unsigned int tile_mem[256];
	for(int j = 0; j < tile_count; j++){
		t_pkt_32 tmp_s;
		tmp_s = tile_s_in.read();
		tile_s_out.write(tmp_s);
		tile_mem[j] = tmp_s.data;
	}

	for(unsigned short itr =  0; itr < 2*count ; itr++){
		for(unsigned short j = 0; j < tile_count; j++){
			#pragma HLS dataflow
			unsigned short offset = tile_mem[j] & 0xffff;
			unsigned short tile_x = tile_mem[j] >> 16;
			process_SLR( in, out, xdim0, offset, tile_x, sizex, sizey);
		}
	}
}
}
