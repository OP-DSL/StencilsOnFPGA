#include "stencil.h"

static void axis2_fifo256(hls::stream <t_pkt> &in, hls::stream<uint256_dt> &out,  const unsigned int total_itr)
{
	for (int itr = 0; itr < total_itr; itr++){
		#pragma HLS PIPELINE II=1
		#pragma HLS loop_tripcount min=min_grid max=max_grid avg=avg_grid
		t_pkt tmp = in.read();
		out << tmp.data;
	}
}

static void fifo256_2axis(hls::stream <uint256_dt> &in, hls::stream<t_pkt> &out, const unsigned int total_itr)
{
	for (int itr = 0; itr < total_itr; itr++){
		#pragma HLS PIPELINE II=1
		#pragma HLS loop_tripcount min=min_grid max=max_grid avg=avg_grid
		t_pkt tmp;
		tmp.data = in.read();
		out.write(tmp);
	}
}

static void inline get_stencil_coefficent(const float & alpha, const float & beta,
		const float & delta_t, const int & init_idx, float * a,
		float * b, float * c)
{
	for (int i = 0; i < VEC_FACTOR; i++)
	{
#pragma HLS UNROLL

		int idx = init_idx + i;
		a[i] = 0.5 * (alpha * std::pow(idx,2) - beta * idx);
		b[i] = 1 - alpha * std::pow(idx,2) - beta;
		c[i] = 0.5 * (alpha * std::pow(idx,2) + beta * idx);
	}
}

static void process_grid(hls::stream<uint256_dt> &rd_buffer, hls::stream<uint256_dt> &wr_buffer, struct data_G data_g,
		const float alpha, const float beta, const float delta_t)
{

	short end_index = data_g.end_index;

    // Registers to hold data specified by stencil
	float row_arr1[VEC_FACTOR + 2];
	float mem_wr[VEC_FACTOR];

    // partioning array into individual registers
	#pragma HLS ARRAY_PARTITION variable=row_arr1 complete dim=1
	#pragma HLS ARRAY_PARTITION variable=mem_wr complete dim=1

	//register to hold coefficients
	float a[VEC_FACTOR];
	float b[VEC_FACTOR];
	float c[VEC_FACTOR];

	//partitioning
	#pragma HLS ARRAY_PARTITION dim=1 type=complete variable=a
	#pragma HLS ARRAY_PARTITION dim=1 type=complete variable=b
	#pragma HLS ARRAY_PARTITION dim=1 type=complete variable=c

    // cyclic buffers to hold larger number of elements. //No need for cyclic buffer
//	uint256_dt row1_n[max_depth_8];
//	uint256_dt row2_n[max_depth_8];
//	uint256_dt row3_n[max_depth_8];
//
//	#pragma HLS RESOURCE variable=row1_n core=XPM_MEMORY uram latency=2
//	#pragma HLS RESOURCE variable=row2_n core=XPM_MEMORY uram latency=2
//	#pragma HLS RESOURCE variable=row3_n core=XPM_MEMORY uram latency=2

	unsigned short sizex = data_g.sizex;
//	unsigned short end_row = data_g.end_row;
//	unsigned short outer_loop_limit = data_g.outer_loop_limit;
	unsigned int grid_size = data_g.gridsize;
	unsigned short end_index_minus1 = data_g.endindex_minus1;
//	unsigned short end_row_plus1 = data_g.endrow_plus1;
//	unsigned short end_row_plus2 = data_g.endrow_plus2;
//	unsigned short end_row_minus1 = data_g.endrow_minus1;
	unsigned int grid_data_size = data_g.total_itr_256;

    uint256_dt tmp1_f1, tmp1, tmp1_b1;
//    uint256_dt tmp1, tmp2, tmp3;
    uint256_dt update_j;

    unsigned int window_delay = 1;

    // flattened loop to reduce the inter loop latency
    unsigned short j;
    unsigned short j_d = -window_delay;
    unsigned int write_itr = 0;

    for(unsigned int itr = 0; itr < grid_size + window_delay; itr++) {
        #pragma HLS loop_tripcount min=min_block_x max=max_block_x avg=avg_block_x
        #pragma HLS PIPELINE II=1

    	j = j_d;

    	bool cmp_j = (j == end_index - 1);

    	if(cmp_j){
    		j_d = 0;
    	} else {
    		j_d++;
    	}

        tmp1_b1 = tmp1;
        tmp1 = tmp1_f1;
        // continuous data-flow for all the grids in the batch
        bool cond_tmp1 = (itr < grid_data_size); // && (j != end_index - 1 + window_delay);

        if(cond_tmp1){
            tmp1_f1 = rd_buffer.read();

//            printf("vector read: %d ={", itr);

            for(int k = 0; k < VEC_FACTOR; k++)
            {
				data_conv tmp;
				tmp.i = tmp1_f1.range(DATATYPE_SIZE * (k + 1) - 1, k * DATATYPE_SIZE);

//	            if (k != VEC_FACTOR - 1)
//	            	printf("%f, ", tmp.f);
//	            else
//	            	printf("%f}\n", tmp.f);
            }
        }

        vec2arr: for(int k = 0; k < VEC_FACTOR; k++){
            data_conv tmp1_u;
            tmp1_u.i = tmp1.range(DATATYPE_SIZE * (k + 1) - 1, k * DATATYPE_SIZE);
//            tmp2_u.i = tmp2.range(DATATYPE_SIZE * (k + 1) - 1, k * DATATYPE_SIZE);
//            tmp3_u.i = tmp3.range(DATATYPE_SIZE * (k + 1) - 1, k * DATATYPE_SIZE);

//            row_arr3[k] =  tmp3_u.f;
//            row_arr2[k+1] = tmp2_u.f;
            row_arr1[k+1] =  tmp1_u.f;
        }
        data_conv tmp1_o1, tmp2_o2;
        tmp1_o1.i = tmp1_b1.range(DATATYPE_SIZE * (VEC_FACTOR) - 1, (VEC_FACTOR-1) * DATATYPE_SIZE);
        tmp2_o2.i = tmp1_f1.range(DATATYPE_SIZE * (0 + 1) - 1, 0 * DATATYPE_SIZE);
        row_arr1[0] = tmp1_o1.f;
        row_arr1[VEC_FACTOR + 1] = tmp2_o2.f;

        int init_idx = (j << SHIFT_BITS);

        readCoeff: get_stencil_coefficent(alpha, beta, delta_t, init_idx, a, b, c);

        // stencil computation
        // this loop will be completely unrolled as parent loop is pipelined
        process: for(short q = 0; q < VEC_FACTOR; q++){
        	int index = (j << SHIFT_BITS) + q;
            float r1 = ( (a[q] * row_arr1[q])  + (b[q] * row_arr1[q+1]) + (c[q] * row_arr1[q+2]) );
            float result  = r1;
            bool change_cond = (index <= 0 || index > sizex);

            if (change_cond)
            	mem_wr[q] = row_arr1[q+1];
            else
            	mem_wr[q] = result;
//            mem_wr[q] = change_cond ? row_arr1[q+1] : result;
        }

        array2vec: for(int k = 0; k < VEC_FACTOR; k++){
            data_conv tmp;
            tmp.f = mem_wr[k];
            update_j.range(DATATYPE_SIZE * (k + 1) - 1, k * DATATYPE_SIZE) = tmp.i;
        }
        // conditional write to stream interface
        bool cond_wr = (itr >= 1) && write_itr < grid_data_size;

        if(cond_wr ) {
            wr_buffer << update_j;
            write_itr++;
        }

    }
}
