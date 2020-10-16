static void axis2_fifo256(hls::stream <t_pkt> &in, hls::stream<uint256_dt> &out, const int tile_x, int size_y){
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


static void process_tile( hls::stream<uint256_dt> &rd_buffer, hls::stream<uint256_dt> &wr_buffer, const int size0, int size1,  const unsigned short offset, struct data_G data_g){
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
		unsigned short i_dum = 0, j_dum =0;
		for(unsigned int itr = 0; itr < grid_size; itr++) {
			#pragma HLS loop_tripcount min=min_block_x max=max_block_x avg=avg_block_x
			#pragma HLS PIPELINE II=1

			bool j_cond = j_dum >= end_index;
			if(j_cond){
				i_dum = i + 1;
				j_dum = 0;
			}

			i = i_dum;
			j = j_dum;

//			if(i >= outer_loop_limit){
//				i = 0;
//				j = 0;
//			}

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
				short index = (j << SHIFT_BITS) + q + offset;
				float r1 = ( (row_arr2[q])  + (row_arr2[q+2]) );

				float r2 = ( row_arr1[q]  + row_arr3[q] );

				float f1 = r1 + r2;
				float f2 = ldexpf(f1, -3);
				float f3 = ldexpf(row_arr2[q+1], -1);//0.5;
				float result  = f2 + f3;
				bool change_cond = (index <= offset || index > size0 || (i == 1) || (i == end_row));
				mem_wr[q] =  change_cond ? row_arr2[q+1] : result;
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
			j_dum++;
		}
//	}
}
