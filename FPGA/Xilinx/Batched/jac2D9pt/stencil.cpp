

static void axis2_fifo256(hls::stream <t_pkt> &in, hls::stream<uint256_dt> &out, struct data_G data_g){
    unsigned int total_itr = data_g.total_itr_256;
	for (int itr = 0; itr < total_itr; itr++){
		#pragma HLS PIPELINE II=1
		#pragma HLS loop_tripcount min=min_grid max=max_grid avg=avg_grid
		t_pkt tmp = in.read();
		out << tmp.data;
	}
}

static void fifo256_2axis(hls::stream <uint256_dt> &in, hls::stream<t_pkt> &out, struct data_G data_g){
	unsigned int total_itr = data_g.total_itr_256;
	for (int itr = 0; itr < total_itr; itr++){
		#pragma HLS PIPELINE II=1
		#pragma HLS loop_tripcount min=min_grid max=max_grid avg=avg_grid
		t_pkt tmp;
		tmp.data = in.read();
		out.write(tmp);
	}
}

static void process_grid( hls::stream<uint256_dt> &rd_buffer, hls::stream<uint256_dt> &wr_buffer, struct data_G data_g){

	short end_index = data_g.end_index;

    // Registers to hold data specified by stencil
	float row_arr0[VEC_FACTOR+2];
	float row_arr1[VEC_FACTOR+2];
	float row_arr2[VEC_FACTOR+2];
	float mem_wr[VEC_FACTOR];

    // partioning array into individual registers
	#pragma HLS ARRAY_PARTITION variable=row_arr0 complete dim=1
	#pragma HLS ARRAY_PARTITION variable=row_arr1 complete dim=1
	#pragma HLS ARRAY_PARTITION variable=row_arr2 complete dim=1
	#pragma HLS ARRAY_PARTITION variable=mem_wr complete dim=1


    // cyclic buffers to hold larger number of elements
	uint256_dt buffer_0[max_depth_8];
	uint256_dt buffer_1[max_depth_8];

	#pragma HLS RESOURCE variable=buffer_0 core=XPM_MEMORY uram latency=2
	#pragma HLS RESOURCE variable=buffer_1 core=XPM_MEMORY uram latency=2

	unsigned short sizex = data_g.sizex;
	unsigned short end_row = data_g.end_row;
	unsigned short outer_loop_limit = data_g.outer_loop_limit;
	unsigned int grid_size = data_g.gridsize;
	unsigned short end_index_minus1 = data_g.endindex_minus1;
	unsigned short end_row_plus1 = data_g.endrow_plus1;
	unsigned short end_row_plus2 = data_g.endrow_plus2;
	unsigned short end_row_minus1 = data_g.endrow_minus1;
	unsigned int grid_data_size = data_g.total_itr_256;

    uint256_dt S_2_2, S_1_2, S_0_2, S_2_1, S_1_1, S_0_1, S_2_0, S_1_0, S_0_0;
    uint256_dt update_j;


    // flattened loop to reduce the inter loop latency
    unsigned short i = 0, j = 0, j_l = 0;
    unsigned short i_d = 0, j_d = -1;
    for(unsigned int itr = 0; itr < grid_size; itr++) {
        #pragma HLS loop_tripcount min=min_block_x max=max_block_x avg=avg_block_x
        #pragma HLS PIPELINE II=1

    	i = i_d;
    	j = j_d;

    	bool cmp_j = (j == end_index -1 );
    	bool cmp_i = (i == outer_loop_limit -1);

    	if(cmp_j){
    		j_d = 0;
    	} else {
    		j_d++;
    	}

    	if(cmp_j && cmp_i){
    		i_d = 1;
    	} else if(cmp_j){
    		i_d++;
    	}

    	S_0_0 = S_1_0;
    	S_1_0 = S_2_0;
    	S_2_0 = buffer_1[j_l];

        S_0_1 = S_1_1;
        buffer_1[j_l] = S_0_1;


        S_1_1 = S_2_1;
        S_2_1 = buffer_0[j_l];


        S_0_2 = S_1_2;
        buffer_0[j_l] = S_0_2;

        S_1_2 = S_2_2;

        // continuous data-flow for all the grids in the batch
        bool cond_tmp1 = (itr < grid_data_size);
        if(cond_tmp1){
            S_2_2 = rd_buffer.read();
        }



        // line buffer
        j_l++;
        if(j_l >= end_index - 2){
            j_l = 0;
        }


        vec2arr: for(int k = 0; k < VEC_FACTOR; k++){
            data_conv tmp0_u, tmp1_u, tmp2_u;
            tmp0_u.i = S_1_0.range(DATATYPE_SIZE * (k + 1) - 1, k * DATATYPE_SIZE);
            tmp1_u.i = S_1_1.range(DATATYPE_SIZE * (k + 1) - 1, k * DATATYPE_SIZE);
            tmp2_u.i = S_1_2.range(DATATYPE_SIZE * (k + 1) - 1, k * DATATYPE_SIZE);

            row_arr0[k+1] =  tmp0_u.f;
            row_arr1[k+1] =  tmp1_u.f;
            row_arr2[k+1] =  tmp2_u.f;
        }

		data_conv tmp0_o1, tmp0_o2;
		tmp0_o1.i = S_0_0.range(DATATYPE_SIZE * (VEC_FACTOR) - 1, (VEC_FACTOR-1) * DATATYPE_SIZE);
		tmp0_o2.i = S_2_0.range(DATATYPE_SIZE * (0 + 1) - 1, 0 * DATATYPE_SIZE);
		row_arr0[0] = tmp0_o1.f;
		row_arr0[VEC_FACTOR + 1] = tmp0_o2.f;

		data_conv tmp1_o1, tmp1_o2;
		tmp1_o1.i = S_0_1.range(DATATYPE_SIZE * (VEC_FACTOR) - 1, (VEC_FACTOR-1) * DATATYPE_SIZE);
		tmp1_o2.i = S_2_1.range(DATATYPE_SIZE * (0 + 1) - 1, 0 * DATATYPE_SIZE);
		row_arr1[0] = tmp1_o1.f;
		row_arr1[VEC_FACTOR + 1] = tmp1_o2.f;

		data_conv tmp2_o1, tmp2_o2;
		tmp2_o1.i = S_0_2.range(DATATYPE_SIZE * (VEC_FACTOR) - 1, (VEC_FACTOR-1) * DATATYPE_SIZE);
		tmp2_o2.i = S_2_2.range(DATATYPE_SIZE * (0 + 1) - 1, 0 * DATATYPE_SIZE);
		row_arr2[0] = tmp2_o1.f;
		row_arr2[VEC_FACTOR + 1] = tmp2_o2.f;




        // stencil computation
        // this loop will be completely unrolled as parent loop is pipelined
        process: for(short q = 0; q < VEC_FACTOR; q++){
			short index = (j << SHIFT_BITS) + q;
			float r1_1 = row_arr0[q] * (-0.07f);
			float r1_2 = row_arr0[q+1] * (-0.06f);
			float r1_3 = row_arr0[q+2] * (-0.05f);

			float r2_1 = row_arr1[q] * (-0.08f);
			float r2_2 = row_arr1[q+1] * (0.36f);
			float r2_3 = row_arr1[q+2] * (-0.04f);

			float r3_1 = row_arr2[q] * (-0.01f);
			float r3_2 = row_arr2[q+1] * (-0.02f);
			float r3_3 = row_arr2[q+2] * (-0.03f);


			float s1 = r1_1 + r1_2;
			float s2 = r1_3 + r2_1;
			float s3 = r2_2 + r2_3;
			float s4 = r3_1 + r3_2;

			float r1 = s1 + s2;
			float r2 = s3 + s4;
			float r = r1 + r2;
			float result  = r1 + r2 + r3_3;
            bool change_cond = (index <= 0 || index > sizex || (i == 1) || (i == end_row));
            mem_wr[q] = change_cond ? row_arr1[q+1] : result;
        }

        array2vec: for(int k = 0; k < VEC_FACTOR; k++){
            data_conv tmp;
            tmp.f = mem_wr[k];
            update_j.range(DATATYPE_SIZE * (k + 1) - 1, k * DATATYPE_SIZE) = tmp.i;
        }
        // conditional write to stream interface
        bool cond_wr = (i >= 1);
        if(cond_wr ) {
            wr_buffer << update_j;
        }

    }
}
