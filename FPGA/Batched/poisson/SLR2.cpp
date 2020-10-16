#include <ap_int.h>
#include <hls_stream.h>
#include <ap_axi_sdata.h>
#include<math.h>

typedef ap_uint<512> uint512_dt;
typedef ap_uint<256> uint256_dt;
typedef ap_axiu<256,0,0,0> t_pkt;

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



static void axis2_fifo256(hls::stream <t_pkt> &in, hls::stream<uint256_dt> &out, const int xdim0_poisson_kernel_stencil, const int base0, int size1, int batches){
	#pragma HLS dataflow
	int end_index = (xdim0_poisson_kernel_stencil >> (SHIFT_BITS));
	int end_row = size1+2;

	for (int itr = 0; itr < end_row * end_index * batches; itr++){
		#pragma HLS PIPELINE II=1
		#pragma HLS loop_tripcount min=min_grid max=max_grid avg=avg_grid
		t_pkt tmp = in.read();
//		uint512_dt data = tmp.data;
//		out << data.range(255,0);
//		out << data.range(511,256);
		out << tmp.data;
	}
}

static void fifo256_2axis(hls::stream <uint256_dt> &in, hls::stream<t_pkt> &out, const int xdim0_poisson_kernel_stencil, const int base0, int size1, int batches){
	#pragma HLS dataflow
	int end_index = (xdim0_poisson_kernel_stencil >> (SHIFT_BITS));
	int end_row = size1+2;

	for (int itr = 0; itr < end_row * end_index* batches; itr++){
		#pragma HLS PIPELINE II=1
		#pragma HLS loop_tripcount min=min_grid max=max_grid avg=avg_grid
		t_pkt tmp;
//		uint512_dt data;
//		data.range(255,0) = in.read();
//		data.range(511,256) = in.read();
		tmp.data = in.read();
		out.write(tmp);
	}
}

static void process_grid( hls::stream<uint256_dt> &rd_buffer, hls::stream<uint256_dt> &wr_buffer, const int size0, int size1,  const int xdim0_poisson_kernel_stencil, struct data_G data_g){
//	#pragma HLS dataflow

	short end_index = data_g.end_index;




	float row_arr3[PORT_WIDTH];
	float row_arr2[PORT_WIDTH + 2];
	float row_arr1[PORT_WIDTH];
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

		uint256_dt tmp2_f1, tmp2_b1;
		uint256_dt tmp1, tmp2, tmp3;
		uint256_dt update_j;

		unsigned short i = 0, j = 0, j_l = 0;
		for(unsigned int itr = 0; itr < grid_size; itr++) {
			#pragma HLS loop_tripcount min=min_block_x max=max_block_x avg=avg_block_x
			#pragma HLS PIPELINE II=1

			if( j >= end_index){
				i++;
				j = 0;
			}

			if(i >= outer_loop_limit){
				i = 0;
				j = 0;
			}

			tmp1 = row2_n[j_l];

			tmp2_b1 = tmp2;
			row2_n[j_l] = tmp2_b1;

			tmp2 = tmp2_f1;
			tmp2_f1 = row1_n[j_l];


			bool cond_tmp1 = (i < end_row);
			if(cond_tmp1){
				tmp3 = rd_buffer.read();
			}
			row1_n[j_l] = tmp3;


			// line buffer
			j_l++;
			if(j_l >= end_index - 1){
				j_l = 0;
			}


			vec2arr: for(int k = 0; k < PORT_WIDTH; k++){
				#pragma HLS loop_tripcount min=port_width max=port_width avg=port_width
				data_conv tmp1_u, tmp2_u, tmp3_u;
				tmp1_u.i = tmp1.range(DATATYPE_SIZE * (k + 1) - 1, k * DATATYPE_SIZE);
				tmp2_u.i = tmp2.range(DATATYPE_SIZE * (k + 1) - 1, k * DATATYPE_SIZE);
				tmp3_u.i = tmp3.range(DATATYPE_SIZE * (k + 1) - 1, k * DATATYPE_SIZE);

				row_arr3[k] =  tmp3_u.f;
				row_arr2[k+1] = tmp2_u.f;
				row_arr1[k] =  tmp1_u.f;
			}
			data_conv tmp1_o1, tmp2_o2;
			tmp1_o1.i = tmp2_b1.range(DATATYPE_SIZE * (PORT_WIDTH) - 1, (PORT_WIDTH-1) * DATATYPE_SIZE);
			tmp2_o2.i = tmp2_f1.range(DATATYPE_SIZE * (0 + 1) - 1, 0 * DATATYPE_SIZE);
			row_arr2[0] = tmp1_o1.f;
			row_arr2[PORT_WIDTH + 1] = tmp2_o2.f;



			process: for(short q = 0; q < PORT_WIDTH; q++){
				#pragma HLS loop_tripcount min=port_width max=port_width avg=port_width
				short index = (j << SHIFT_BITS) + q;
				float r1 = ( (row_arr2[q])  + (row_arr2[q+2]) );

				float r2 = ( row_arr1[q]  + row_arr3[q] );

				float f1 = r1 + r2;
				float f2 = ldexpf(f1, -3);
				float f3 = ldexpf(row_arr2[q+1], -1);//0.5;
				float result  = f2 + f3;
				bool change_cond = (index <= 0 || index > size0 || (i == 1) || (i == end_row));
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

	#pragma HLS dataflow
    axis2_fifo256(in, streamArray[0], xdim0_poisson_kernel_stencil, base0, size1, batches);

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

//	process_a_row( streamArray[20], streamArray[21], size0, size1, xdim0_poisson_kernel_stencil, data_g);
//	process_a_row( streamArray[21], streamArray[22], size0, size1, xdim0_poisson_kernel_stencil, data_g);
//	process_a_row( streamArray[22], streamArray[23], size0, size1, xdim0_poisson_kernel_stencil, data_g);
//	process_a_row( streamArray[23], streamArray[24], size0, size1, xdim0_poisson_kernel_stencil, data_g);

	fifo256_2axis(streamArray[20], out, xdim0_poisson_kernel_stencil, base0, size1, batches);


}

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



	for(int i =  0; i < 2*count ; i++){
		#pragma HLS dataflow
		process_SLR( in, out, xdim0_poisson_kernel_stencil, base0, xdim1_poisson_kernel_stencil, base1, size0, size1, batches);
	}

}
}
