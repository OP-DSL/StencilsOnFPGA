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



static void read_tile(uint512_dt*  arg0, hls::stream<uint512_dt> &rd_buffer, const int xdim0, const int offset, int tile_x, int size_y, unsigned short start){
	#pragma HLS dataflow
	#pragma HLS stable variable=arg0
	unsigned short end_index = (tile_x >> (SHIFT_BITS+1));
	unsigned short  end_row = size_y+2;
	for(unsigned short i = start; i < end_row; i = i+2){
		unsigned int base_index = (offset + xdim0*i) >> (SHIFT_BITS+1);
		for (unsigned short j = 0; j < end_index; j++){
			#pragma HLS PIPELINE II=1
			rd_buffer << arg0[base_index + j];
		}
	}
}

static void combine_tile(hls::stream<uint512_dt> &rd_buffer0, hls::stream<uint512_dt> &rd_buffer1, hls::stream<uint512_dt> &rd_buffer2, hls::stream<uint512_dt> &rd_buffer3,
		hls::stream<uint512_dt> &combined_buffer, const int xdim0, const int offset, int tile_x, int size_y){
	#pragma HLS dataflow
	unsigned short end_index = (tile_x >> (SHIFT_BITS+1));
	unsigned short  end_row = size_y+2;
	for(unsigned short i = 0; i < end_row; i = i+2){
		for (unsigned short j = 0; j < end_index; j++){
			#pragma HLS PIPELINE II=1
			combined_buffer << rd_buffer0.read();
		}

		for (unsigned short j = 0; j < end_index; j++){
			#pragma HLS PIPELINE II=1
			combined_buffer << rd_buffer1.read();
		}

//		for (unsigned short j = 0; j < end_index; j++){
//			#pragma HLS PIPELINE II=1
//			combined_buffer << rd_buffer2.read();
//		}
//
//		for (unsigned short j = 0; j < end_index; j++){
//			#pragma HLS PIPELINE II=1
//			combined_buffer << rd_buffer3.read();
//		}
	}
}


static void stream_convert_512_256(hls::stream<uint512_dt> &in, hls::stream<uint256_dt> &out, const int tile_x, int size_y){
	#pragma HLS dataflow
	unsigned short end_index = (tile_x >> (SHIFT_BITS+1));
	unsigned short end_row = size_y+2;
	unsigned int tot_itr = end_row * end_index;
	for (int itr = 0; itr < tot_itr; itr++){
		#pragma HLS PIPELINE II=2
		#pragma HLS loop_tripcount min=min_grid max=max_grid avg=avg_grid
		uint512_dt tmp = in.read();
		uint256_dt var_l = tmp.range(255,0);
		uint256_dt var_h = tmp.range(511,256);;
		out << var_l;
		out << var_h;
	}
}

static void stream_convert_256_512(hls::stream<uint256_dt> &in, hls::stream<uint512_dt> &out, const int tile_x, int size_y, unsigned short offset){
	#pragma HLS dataflow
	unsigned short adjust = (offset == 0)? 0: 2;
	unsigned short end_index = (tile_x >> (SHIFT_BITS+1));
	unsigned short end_row = size_y+2;
	unsigned int tot_itr = end_row * end_index;
	unsigned short j = 0;
	for (int itr = 0; itr < tot_itr; itr++){
		#pragma HLS PIPELINE II=2
		#pragma HLS loop_tripcount min=min_grid max=max_grid avg=avg_grid
		uint512_dt tmp;
		uint256_dt tmp0_in = in.read();
		uint256_dt tmp1_in = in.read();
		bool cond = (j == end_index);
		if(cond){
			j = 0;
		}
		tmp.range(255,0) = tmp0_in;
		tmp.range(511,256) = tmp1_in;
		bool skip = offset != 0 && (j == 0 || j == 1);
		if(!skip){
			out << tmp;
		}
		j++;
	}
}

static void axis2_fifo256(hls::stream <t_pkt> &in, hls::stream<uint256_dt> &out, const int tile_x, int size_y){
	#pragma HLS dataflow
	unsigned short end_index = (tile_x >> (SHIFT_BITS));
	unsigned short end_row = size_y+2;
	unsigned int tot_itr = end_row * end_index;
	for (int itr = 0; itr < tot_itr; itr++){
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
	for (int itr = 0; itr < tot_itr; itr++){
		#pragma HLS PIPELINE II=1
		#pragma HLS loop_tripcount min=min_grid max=max_grid avg=avg_grid
		t_pkt tmp;
		tmp.data = in.read();
		out.write(tmp);
	}
}

static void process_tile( hls::stream<uint256_dt> &rd_buffer, hls::stream<uint256_dt> &wr_buffer,
		const unsigned short size_x, const unsigned short size_y,  const unsigned short offset, struct data_G data_g){
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

			bool cmp1 = j >= end_index;
			bool cmp2 = i >= outer_loop_limit - 1 && cmp1;
			short i_1 = i, j_1 = j;
//			if( cmp1 ){
//				i_1++;
//				j_1 = 0;
//			}

			if(cmp2){
				i_1 = 0;
				j_1 = -1;
			} else if(cmp1){
				i_1 = i+1;
				j_1 = 0;
			}
			i = i_1;
			j = j_1;



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


				bool change_cond = (index <= offset || index > size_x || (i == 1) || (i == end_row));
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

static void separate_tile(hls::stream<uint512_dt> &in_buffer, hls::stream<uint512_dt> &wr_buffer0, hls::stream<uint512_dt> &wr_buffer1, hls::stream<uint512_dt> &wr_buffer2,
		hls::stream<uint512_dt> &wr_buffer3, const int xdim0, const int offset, int tile_x, int size_y){
	#pragma HLS dataflow
	unsigned short full_size = (tile_x >> (SHIFT_BITS+1));
	unsigned short adjust = offset == 0 ? 0 : 2;
	unsigned short end_index = full_size-adjust;
	unsigned short end_row = size_y + 2;
	for(unsigned short i = 0; i < end_row; i = i+2){
		for (unsigned short j = 0; j < end_index; j++){
			#pragma HLS PIPELINE II=1
			wr_buffer0 << in_buffer.read();
		}

		for (unsigned short j = 0; j < end_index; j++){
			#pragma HLS PIPELINE II=1
			wr_buffer1 << in_buffer.read();
		}
//
//		for (unsigned short j = 0; j < end_index; j++){
//			#pragma HLS PIPELINE II=1
//			wr_buffer2 << in_buffer.read();
//		}
//
//		for (unsigned short j = 0; j < end_index; j++){
//			#pragma HLS PIPELINE II=1
//			wr_buffer3 << in_buffer.read();
//		}
	}
}

static void write_tile( uint512_dt*  arg1, hls::stream<uint512_dt> &wr_buffer, const int xdim0, const unsigned short offset,
			unsigned short tile_x, unsigned short size_y, unsigned short start){
	#pragma HLS dataflow
	#pragma HLS stable variable=arg1
	unsigned short full_size = (tile_x >> (SHIFT_BITS+1));
	unsigned short adjust = offset == 0 ? 0 : 2;
	unsigned short end_index = full_size-adjust;
	unsigned short end_row = size_y+2;
	for(unsigned short i = start; i < end_row; i = i + 2){
		int base_index = ((offset + xdim0*i) >> (SHIFT_BITS+1)) + adjust;
		for (unsigned short j = 0; j < end_index; j++){
			#pragma HLS PIPELINE II=1
			arg1[base_index + j] = wr_buffer.read();
		}
	}
}


void process_SLR0 (uint512_dt*  arg0_0, uint512_dt*  arg0_1,/* uint512_dt*  arg0_2, uint512_dt*  arg0_3,*/
				   uint512_dt*  arg1_0,  uint512_dt*  arg1_1,/* uint512_dt*  arg1_2, uint512_dt*  arg1_3,*/
				   hls::stream <t_pkt> &in, hls::stream <t_pkt> &out,
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
	data_g.gridsize = (data_g.outer_loop_limit * data_g.end_index +1);
	data_g.endindex_minus1 = data_g.end_index -1;
	data_g.endrow_plus1 = data_g.end_row + 1;
	data_g.endrow_plus2 = data_g.end_row + 2;
	data_g.endrow_minus1 = data_g.end_row - 1;



	#pragma HLS dataflow
//	#pragma HLS stable variable=arg0_0
//	#pragma HLS stable variable=arg0_1
//	#pragma HLS stable variable=arg1_0
//	#pragma HLS stable variable=arg1_1

	read_tile(arg0_0, rd_bufferArr[0], xdim0, offset, tile_x, size_y, 0);
	read_tile(arg0_1, rd_bufferArr[1], xdim0, offset, tile_x, size_y, 1);
//	read_tile(arg0_2, rd_bufferArr[2], xdim0, offset, tile_x, size_y, 2);
//	read_tile(arg0_3, rd_bufferArr[3], xdim0, offset, tile_x, size_y, 3);

	combine_tile(rd_bufferArr[0], rd_bufferArr[1], rd_bufferArr[2], rd_bufferArr[3], streamArray_512[0], xdim0, offset, tile_x, size_y);
	stream_convert_512_256(streamArray_512[0], streamArray[0], tile_x, size_y);

	process_tile( streamArray[0], streamArray[1], size_x, size_y, offset, data_g);
//	process_tile( streamArray[1], streamArray[2], size_x, size_y, offset, data_g);
//	process_tile( streamArray[2], streamArray[3], size_x, size_y, offset, data_g);
//	process_tile( streamArray[3], streamArray[4], size_x, size_y, offset, data_g);
//
//	process_tile( streamArray[4], streamArray[5], size_x, size_y, offset, data_g);
//	process_tile( streamArray[5], streamArray[6], size_x, size_y, offset, data_g);
//	process_tile( streamArray[6], streamArray[7], size_x, size_y, offset, data_g);
//	process_tile( streamArray[7], streamArray[8], size_x, size_y, offset, data_g);
//
//	process_tile( streamArray[8], streamArray[9], size_x, size_y, offset, data_g);
//	process_grid( streamArray[9], streamArray[10], size0, size1, xdim0_poisson_kernel_stencil,  data_g);
//	process_grid( streamArray[10], streamArray[11], size0, size1, xdim0_poisson_kernel_stencil, data_g);
//	process_grid( streamArray[11], streamArray[12], size0, size1, xdim0_poisson_kernel_stencil, data_g);

//	process_grid( streamArray[12], streamArray[13], size0, size1, xdim0_poisson_kernel_stencil, data_g);
//	process_grid( streamArray[13], streamArray[14], size0, size1, xdim0_poisson_kernel_stencil, data_g);
//	process_grid( streamArray[14], streamArray[15], size0, size1, xdim0_poisson_kernel_stencil, data_g);
//	process_grid( streamArray[15], streamArray[16], size0, size1, xdim0_poisson_kernel_stencil, data_g);

//	process_grid( streamArray[16], streamArray[17], size0, size1, xdim0_poisson_kernel_stencil, data_g);
//	process_grid( streamArray[17], streamArray[18], size0, size1, xdim0_poisson_kernel_stencil, data_g);
//	process_grid( streamArray[18], streamArray[19], size0, size1, xdim0_poisson_kernel_stencil, data_g);
//	process_grid( streamArray[19], streamArray[20], size0, size1, xdim0_poisson_kernel_stencil, data_g);

	fifo256_2axis(streamArray[1], out, tile_x, size_y);
	axis2_fifo256(in, streamArray[31], tile_x, size_y);



	stream_convert_256_512(streamArray[31], streamArray_512[1], tile_x, size_y, offset);

	separate_tile(streamArray_512[1], wr_bufferArr[0], wr_bufferArr[1], wr_bufferArr[2], wr_bufferArr[3], xdim0, offset, tile_x, size_y);
	write_tile(arg1_0, wr_bufferArr[0], xdim0, offset, tile_x, size_y, 0);
	write_tile(arg1_1, wr_bufferArr[1], xdim0, offset, tile_x, size_y, 1);
//	write_tile(arg1_2, wr_bufferArr[2], xdim0, offset, tile_x, size_y, 2);
//	write_tile(arg1_3, wr_bufferArr[3], xdim0, offset, tile_x, size_y, 3);

}


//-DHOST_CODE_OPT -DLOCAL_BUF_OPT -DDF_OPT -DFP_OPT
//--sc stencil_SLR0_1.out:stencil_SLR1_1.in
//--sc stencil_SLR1_1.out:stencil_SLR0_1.in

extern "C" {
void stencil_SLR0(
		uint512_dt*  arg0_0,
		uint512_dt*  arg0_1,
//		uint512_dt*  arg0_2,
//		uint512_dt*  arg0_3,

		uint512_dt*  arg1_0,
		uint512_dt*  arg1_1,
//		uint512_dt*  arg1_2,
//		uint512_dt*  arg1_3,

		const unsigned int* tile,
		const int tile_count,
		const int sizex,
		const int sizey,
		const int xdim0,
		const int count,

		hls::stream <t_pkt_32> &tile_s_out,
		hls::stream <t_pkt> &in,
		hls::stream <t_pkt> &out){

	#pragma HLS INTERFACE depth=4096 m_axi port = arg0_0 offset = slave bundle = gmem0 max_read_burst_length=64 max_write_burst_length=64 //num_read_outstanding=2 num_write_outstanding=2
	#pragma HLS INTERFACE depth=4096 m_axi port = arg0_1 offset = slave bundle = gmem1 max_read_burst_length=64 max_write_burst_length=64
//	#pragma HLS INTERFACE depth=4096 m_axi port = arg0_2 offset = slave bundle = gmem0
//	#pragma HLS INTERFACE depth=4096 m_axi port = arg0_3 offset = slave bundle = gmem0

	#pragma HLS INTERFACE depth=4096 m_axi port = arg1_0 offset = slave bundle = gmem0
	#pragma HLS INTERFACE depth=4096 m_axi port = arg1_1 offset = slave bundle = gmem1
//	#pragma HLS INTERFACE depth=4096 m_axi port = arg1_2 offset = slave bundle = gmem0
//	#pragma HLS INTERFACE depth=4096 m_axi port = arg1_3 offset = slave bundle = gmem0


	#pragma HLS INTERFACE depth=4096 m_axi port = tile offset = slave bundle = gmem2

	#pragma HLS INTERFACE s_axilite port = arg0_0 bundle = control
	#pragma HLS INTERFACE s_axilite port = arg0_1 bundle = control
//	#pragma HLS INTERFACE s_axilite port = arg0_2 bundle = control
//	#pragma HLS INTERFACE s_axilite port = arg0_3 bundle = control

	#pragma HLS INTERFACE s_axilite port = arg1_0 bundle = control
	#pragma HLS INTERFACE s_axilite port = arg1_1 bundle = control
//	#pragma HLS INTERFACE s_axilite port = arg1_2 bundle = control
//	#pragma HLS INTERFACE s_axilite port = arg1_3 bundle = control

	#pragma HLS INTERFACE s_axilite port = tile bundle = control
	#pragma HLS INTERFACE axis port = tile_s_out  register
	#pragma HLS INTERFACE axis port = in  register
	#pragma HLS INTERFACE axis port = out register

	#pragma HLS INTERFACE s_axilite port = tile_count bundle = control
	#pragma HLS INTERFACE s_axilite port = sizex bundle = control
	#pragma HLS INTERFACE s_axilite port = sizey bundle = control
	#pragma HLS INTERFACE s_axilite port = xdim0 bundle = control
	#pragma HLS INTERFACE s_axilite port = count bundle = control
	#pragma HLS INTERFACE s_axilite port = return bundle = control

//	#pragma HLS stable variable=arg0_0
//	#pragma HLS stable variable=arg0_1
//	#pragma HLS stable variable=arg1_0
//	#pragma HLS stable variable=arg1_1


	unsigned int tile_mem[256];
	for(int j = 0; j < tile_count; j++){
		#pragma HLS PIPELINE II=1
		t_pkt_32 tmp_s;
		unsigned int tmp;
		tmp = tile[j];
		tile_mem[j] = tmp;
		tmp_s.data = tmp;
		tile_s_out.write(tmp_s);
	}

	for(int i =  0; i < count; i++){
		for(int j = 0; j < tile_count; j++){
			unsigned short offset = tile_mem[j] & 0xffff;
			unsigned short tile_x = tile_mem[j] >> 16;
			process_SLR0(arg0_0, arg0_1/*, arg0_2, arg0_3*/, arg1_0, arg1_1/*, arg1_2, arg1_3*/, in, out, xdim0, offset, tile_x, sizex, sizey);
		}

		for(int j = 0; j < tile_count; j++){
			unsigned short offset = tile_mem[j] & 0xffff;
			unsigned short tile_x = tile_mem[j] >> 16;
			process_SLR0(arg1_0, arg1_1/*, arg0_2, arg0_3*/, arg0_0, arg0_1/*, arg1_2, arg1_3*/, in, out, xdim0, offset, tile_x, sizex, sizey);
		}
	}
}
}
