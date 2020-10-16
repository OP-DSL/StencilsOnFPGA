#include <ap_int.h>
#include <hls_stream.h>
#include <ap_axi_sdata.h>
#include<math.h>

typedef ap_uint<512> uint512_dt;
typedef ap_uint<256> uint256_dt;
typedef ap_axiu<256,0,0,0> t_pkt;
typedef ap_axiu<32,0,0,0> t_pkt_32;

#define MAX_SIZE_X 8192
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

typedef union  {
   int i;
   float f;
} data_conv;

struct data_G{
	unsigned short end_index;
	unsigned short end_row;
	unsigned int gridsize;
	unsigned short outer_loop_limit;
	unsigned short endrow_plus2;
	unsigned short endrow_plus1;
	unsigned short endrow_minus1;
	unsigned short endindex_minus1;
};



static void axis2_fifo256(hls::stream <t_pkt> &in, hls::stream<uint256_dt> &out, const int tile_x, int size_y){
	#pragma HLS dataflow
	unsigned short end_index = (tile_x >> (SHIFT_BITS));
	unsigned short end_row = size_y+2;
	unsigned int tot_itr = end_row * end_index;

	for (unsigned int itr = 0; itr < tot_itr; itr++){
		#pragma HLS PIPELINE II=1
		#pragma HLS loop_tripcount min=min_grid max=max_grid avg=avg_grid
		t_pkt tmp = in.read();
		out << tmp.data;
	}
}

static void fifo256_2axis(hls::stream <uint256_dt> &in, hls::stream<t_pkt> &out, const int tile_x, int size_y){
	#pragma HLS dataflow
	unsigned short end_index = (tile_x >> (SHIFT_BITS));
	unsigned short end_row = size_y+2;
	unsigned int tot_itr = end_row * end_index;

	for (unsigned int itr = 0; itr < tot_itr; itr++){
		#pragma HLS PIPELINE II=1
		#pragma HLS loop_tripcount min=min_grid max=max_grid avg=avg_grid
		t_pkt tmp;
		tmp.data = in.read();
		out.write(tmp);
	}
}

static void process_tile( hls::stream<uint256_dt> &rd_buffer, hls::stream<uint256_dt> &wr_buffer,
		const unsigned short size0, const unsigned short size1,  const unsigned short offset, struct data_G data_g){
//	#pragma HLS dataflow

	short end_index = data_g.end_index;




	float row_arr3[PORT_WIDTH + 2];
	float row_arr2[PORT_WIDTH + 2];
	float row_arr1[PORT_WIDTH + 2];

	float same_val[PORT_WIDTH];
	float mem_wr[PORT_WIDTH];

	#pragma HLS ARRAY_PARTITION variable=row_arr3 complete dim=1
	#pragma HLS ARRAY_PARTITION variable=row_arr2 complete dim=1
	#pragma HLS ARRAY_PARTITION variable=row_arr1 complete dim=1
	#pragma HLS ARRAY_PARTITION variable=mem_wr complete dim=1



	uint256_dt row1_n[max_depth_8];
	uint256_dt row2_n[max_depth_8];
	uint256_dt row3_n[max_depth_8];

	#pragma HLS RESOURCE variable=row1_n core=XPM_MEMORY uram latency=2
	#pragma HLS RESOURCE variable=row2_n core=XPM_MEMORY uram latency=2
	#pragma HLS RESOURCE variable=row3_n core=XPM_MEMORY uram latency=2

	unsigned short end_row = data_g.end_row;
	unsigned short outer_loop_limit = data_g.outer_loop_limit;
	unsigned int grid_size = data_g.gridsize;
	unsigned short end_index_minus1 = data_g.endindex_minus1;
	unsigned short end_row_plus1 = data_g.endrow_plus1;
	unsigned short end_row_plus2 = data_g.endrow_plus2;
	unsigned short end_row_minus1 = data_g.endrow_minus1;

		uint256_dt tmp1_f1, tmp1_b1, tmp2_f1, tmp2_b1, tmp3_f1, tmp3_b1;
		uint256_dt tmp1, tmp2, tmp3;
		uint256_dt update_j;

		short i = 0, j = -1, j_l = 0;
		for(unsigned int itr = 0; itr < grid_size; itr++) {
			#pragma HLS loop_tripcount min=min_block_x max=max_block_x avg=avg_block_x
			#pragma HLS PIPELINE II=1

			bool i_cond = j >= end_index;
			if(i_cond){
				i++;
				j = 0;
			}

			bool j_cond = i >= outer_loop_limit;
			if(j_cond){
				i = 0;
				j = -1;
			}

			tmp1_b1 = tmp1;
			tmp1 = tmp1_f1;
			tmp1_f1 = row2_n[j_l];

			tmp2_b1 = tmp2;
			row2_n[j_l] = tmp2_b1;

			tmp2 = tmp2_f1;
			tmp2_f1 = row3_n[j_l];


			tmp3_b1 = tmp3;
			row3_n[j_l] = tmp3_b1;
			tmp3 = tmp3_f1;

			bool skip_cond = (i == end_row -1 && j == end_index -1);
			bool cond_tmp1 = (i < end_row && !skip_cond );
			if(cond_tmp1){
				tmp3_f1 = rd_buffer.read();
			}



			// line buffer
			j_l++;
			if(j_l >= end_index - 2){
				j_l = 0;
			}


			vec2arr: for(int k = 0; k < PORT_WIDTH; k++){
				#pragma HLS loop_tripcount min=port_width max=port_width avg=port_width
				data_conv tmp1_u, tmp2_u, tmp3_u, tmp4_u;
				tmp1_u.i = tmp1.range(DATATYPE_SIZE * (k + 1) - 1, k * DATATYPE_SIZE);
				tmp2_u.i = tmp2.range(DATATYPE_SIZE * (k + 1) - 1, k * DATATYPE_SIZE);
				tmp3_u.i = tmp3.range(DATATYPE_SIZE * (k + 1) - 1, k * DATATYPE_SIZE);
				tmp4_u.i = tmp2_f1.range(DATATYPE_SIZE * (k + 1) - 1, k * DATATYPE_SIZE);

				row_arr3[k+1] =  tmp3_u.f;
				row_arr2[k+1] =  tmp2_u.f;
				row_arr1[k+1] =  tmp1_u.f;
				same_val[k] = tmp4_u.f;
			}

			data_conv tmp2_o1, tmp2_o2;
			tmp2_o1.i = tmp2_b1.range(DATATYPE_SIZE * (PORT_WIDTH) - 1, (PORT_WIDTH-1) * DATATYPE_SIZE);
			tmp2_o2.i = tmp2_f1.range(DATATYPE_SIZE * (0 + 1) - 1, 0 * DATATYPE_SIZE);
			row_arr2[0] = tmp2_o1.f;
			row_arr2[PORT_WIDTH + 1] = tmp2_o2.f;

			data_conv tmp1_o1, tmp1_o2;
			tmp1_o1.i = tmp1_b1.range(DATATYPE_SIZE * (PORT_WIDTH) - 1, (PORT_WIDTH-1) * DATATYPE_SIZE);
			tmp1_o2.i = tmp1_f1.range(DATATYPE_SIZE * (0 + 1) - 1, 0 * DATATYPE_SIZE);
			row_arr1[0] = tmp1_o1.f;
			row_arr1[PORT_WIDTH + 1] = tmp1_o2.f;

			data_conv tmp3_o1, tmp3_o2;
			tmp3_o1.i = tmp3_b1.range(DATATYPE_SIZE * (PORT_WIDTH) - 1, (PORT_WIDTH-1) * DATATYPE_SIZE);
			tmp3_o2.i = tmp3_f1.range(DATATYPE_SIZE * (0 + 1) - 1, 0 * DATATYPE_SIZE);
			row_arr3[0] = tmp3_o1.f;
			row_arr3[PORT_WIDTH + 1] = tmp3_o2.f;



			process: for(short q = 0; q < PORT_WIDTH; q++){
				#pragma HLS loop_tripcount min=port_width max=port_width avg=port_width
				unsigned short index = (j << SHIFT_BITS) + q + offset;
				float r1_1 = row_arr1[q] * (-0.07f);
				float r1_2 = row_arr1[q+1] * (-0.06f);
				float r1_3 = row_arr1[q+2] * (-0.05f);
//				float r1 =  r1_1 + r1_2 + r1_3 ;

				float r2_1 = row_arr2[q] * (-0.08f);
				float r2_2 = row_arr2[q+1] * (0.36f);
				float r2_3 = row_arr2[q+2] * (-0.04f);
//				float r2 = r2_1  + r2_2  + r2_3;

				float r3_1 = row_arr3[q] * (-0.01f);
				float r3_2 = row_arr3[q+1] * (-0.02f);
				float r3_3 = row_arr3[q+2] * (-0.03f);
//				float r3 =  r3_1 + r3_2 + r3_3;


				float s1 = r1_1 + r1_2;
				float s2 = r1_3 + r2_1;
				float s3 = r2_2 + r2_3;
				float s4 = r3_1 + r3_2;

				float r1 = s1 + s2;
				float r2 = s3 + s4;
				float r = r1 + r2;
				float result  = r1 + r2 + r3_3;

//				#pragma HLS RESOURCE variable=r1_1 core=FMul_meddsp
//				#pragma HLS RESOURCE variable=r1_2 core=FMul_meddsp
//				#pragma HLS RESOURCE variable=r1_3 core=FMul_meddsp

//				#pragma HLS RESOURCE variable=r2_1 core=FMul_meddsp
//				#pragma HLS RESOURCE variable=r2_2 core=FMul_meddsp
//				#pragma HLS RESOURCE variable=r2_3 core=FMul_meddsp

//				#pragma HLS RESOURCE variable=r3_1 core=FMul_meddsp
//				#pragma HLS RESOURCE variable=r3_2 core=FMul_meddsp
//				#pragma HLS RESOURCE variable=r3_3 core=FMul_meddsp



				#pragma HLS RESOURCE variable=s1 core=FAddSub_nodsp
				#pragma HLS RESOURCE variable=s2 core=FAddSub_nodsp
				#pragma HLS RESOURCE variable=s3 core=FAddSub_nodsp
				#pragma HLS RESOURCE variable=s4 core=FAddSub_nodsp

				#pragma HLS RESOURCE variable=r1 core=FAddSub_nodsp
//				#pragma HLS RESOURCE variable=r2 core=FAddSub_nodsp
//				#pragma HLS RESOURCE variable=r core=FAddSub_nodsp
//				#pragma HLS RESOURCE variable=result core=FAddSub_nodsp


				bool change_cond = (index <= offset || index > size0 || (i == 1) || (i == end_row));
				mem_wr[q] = change_cond ? row_arr2[q+1] : result;
			}


			array2vec: for(int k = 0; k < PORT_WIDTH; k++){
				#pragma HLS loop_tripcount min=port_width max=port_width avg=port_width
				data_conv tmp;
				tmp.f = mem_wr[k];
				update_j.range(DATATYPE_SIZE * (k + 1) - 1, k * DATATYPE_SIZE) = tmp.i;
			}
			bool cond_wr = (i >= 1);
			if(cond_wr ) {
				wr_buffer << update_j;
			}

			j++;
		}
//	}
}
//-DHOST_CODE_OPT -DLOCAL_BUF_OPT -DDF_OPT -DFP_OPT
//--sc ops_poisson_kernel_stencil_SLR0.out:ops_poisson_kernel_stencil_SLR1.in
//--sc ops_poisson_kernel_stencil_SLR0.in:ops_poisson_kernel_stencil_SLR1.out


//--------------------------------------------------------------------------------------------------------------------------
//-------------------------------------------------------- SLR crossing SLR1 -----------------------------------------------
//--------------------------------------------------------------------------------------------------------------------------




void process_SLR (hls::stream <t_pkt> &in1, hls::stream <t_pkt> &out1, hls::stream <t_pkt> &in2, hls::stream <t_pkt> &out2,
		const int xdim0, const unsigned short offset, const unsigned short tile_x, const unsigned short size_x, const unsigned short size_y){


    static hls::stream<uint256_dt> streamArray[40 + 1];
    #pragma HLS STREAM variable = streamArray depth = 2

    struct data_G data_g;
	data_g.end_index = (tile_x >> SHIFT_BITS);
	data_g.end_row = size_y+2;
	data_g.outer_loop_limit = size_y+3;
	data_g.gridsize =  (data_g.outer_loop_limit * data_g.end_index +1);
	data_g.endindex_minus1 = data_g.end_index -1;
	data_g.endrow_plus1 = data_g.end_row + 1;
	data_g.endrow_plus2 = data_g.end_row + 2;
	data_g.endrow_minus1 = data_g.end_row - 1;



	#pragma HLS dataflow
    axis2_fifo256(in1, streamArray[0], tile_x, size_y);

    process_tile( streamArray[0], streamArray[1], size_x, size_y, offset, data_g);
//    process_tile( streamArray[1], streamArray[2], size_x, size_y, offset, data_g);
//    process_tile( streamArray[2], streamArray[3], size_x, size_y, offset, data_g);
//    process_tile( streamArray[3], streamArray[4], size_x, size_y, offset, data_g);
////
//    process_tile( streamArray[4], streamArray[5], size_x, size_y, offset, data_g);
//    process_grid( streamArray[5], streamArray[6], size0, size1, xdim0_poisson_kernel_stencil, data_g);
//    process_grid( streamArray[6], streamArray[7], size0, size1, xdim0_poisson_kernel_stencil, data_g);
//    process_grid( streamArray[7], streamArray[8], size0, size1, xdim0_poisson_kernel_stencil, data_g);
//
//	process_a_row( streamArray[8], streamArray[9], size0, size1, xdim0_poisson_kernel_stencil, data_g);
//	process_a_row( streamArray[9], streamArray[10], size0, size1, xdim0_poisson_kernel_stencil, data_g);
//	process_a_row( streamArray[10], streamArray[11], size0, size1, xdim0_poisson_kernel_stencil, data_g);
//	process_a_row( streamArray[11], streamArray[12], size0, size1, xdim0_poisson_kernel_stencil, data_g);

	fifo256_2axis(streamArray[1], out1, tile_x, size_y);
	axis2_fifo256(in2, streamArray[40], tile_x, size_y);

//	process_tile( streamArray[40], streamArray[21], size_x, size_y, offset, data_g);
//	process_tile( streamArray[21], streamArray[22], size_x, size_y, offset, data_g);
//	process_tile( streamArray[22], streamArray[23], size_x, size_y, offset, data_g);
//	process_tile( streamArray[23], streamArray[24], size_x, size_y, offset, data_g);
//
//	process_grid( streamArray[24], streamArray[25], size0, size1, xdim0_poisson_kernel_stencil, data_g);
//	process_grid( streamArray[25], streamArray[26], size0, size1, xdim0_poisson_kernel_stencil, data_g);
//	process_grid( streamArray[26], streamArray[27], size0, size1, xdim0_poisson_kernel_stencil, data_g);
//	process_grid( streamArray[27], streamArray[28], size0, size1, xdim0_poisson_kernel_stencil, data_g);
//
//
//	process_grid( streamArray[28], streamArray[29], size0, size1, xdim0_poisson_kernel_stencil, data_g);
//	process_grid( streamArray[29], streamArray[30], size0, size1, xdim0_poisson_kernel_stencil, data_g);
//	process_grid( streamArray[30], streamArray[31], size0, size1, xdim0_poisson_kernel_stencil, data_g);
//	process_grid( streamArray[31], streamArray[32], size0, size1, xdim0_poisson_kernel_stencil, data_g);

	fifo256_2axis(streamArray[40], out2, tile_x, size_y);


}

extern "C" {
void stencil_SLR1(
		const int tile_count,
		const int sizex,
		const int sizey,
		const int xdim0,
		const int count,

		hls::stream <t_pkt_32> &tile_s_in,
		hls::stream <t_pkt_32> &tile_s_out,

		hls::stream <t_pkt> &in1,
		hls::stream <t_pkt> &out1,
		hls::stream <t_pkt> &in2,
		hls::stream <t_pkt> &out2
		){

	#pragma HLS INTERFACE axis port = in1 register
	#pragma HLS INTERFACE axis port = out1 register
	#pragma HLS INTERFACE axis port = in2 register
	#pragma HLS INTERFACE axis port = out2 register
	#pragma HLS INTERFACE axis port = tile_s_in register
	#pragma HLS INTERFACE axis port = tile_s_out register

	#pragma HLS INTERFACE s_axilite port = tile_count bundle = control
	#pragma HLS INTERFACE s_axilite port = sizey bundle = control
	#pragma HLS INTERFACE s_axilite port = sizex bundle = control
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

	for(int i =  0; i < 2*count ; i++){
		#pragma HLS dataflow
		for(int j = 0; j < tile_count; j++){
			unsigned short offset = tile_mem[j] & 0xffff;
			unsigned short tile_x = tile_mem[j] >> 16;
			process_SLR( in1, out1,in2, out2, xdim0, offset, tile_x, sizex, sizey);
		}
	}

}
}
