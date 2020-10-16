template <typename T>
static T register_it(T x){
	#pragma HLS inline off
	T tmp = x;
	return tmp;
}


static void axis2_fifo256(hls::stream <t_pkt> &in, hls::stream<uint256_dt> &out, unsigned int total_itr){
//	unsigned short end_index = (tile_x >> (SHIFT_BITS));
//	unsigned short end_row = size_y+2;
//	unsigned int tot_itr = end_row * end_index;
	for (unsigned int itr = 0; itr < total_itr; itr++){
		#pragma HLS PIPELINE II=1
		#pragma HLS loop_tripcount min=min_grid max=max_grid avg=avg_grid
		t_pkt tmp = in.read();
		out << tmp.data;
	}
}

static void fifo256_2axis(hls::stream <uint256_dt> &in, hls::stream<t_pkt> &out, unsigned int total_itr){
//	unsigned short end_index = (tile_x >> (SHIFT_BITS));
//	unsigned short end_row = size_y+2;
//	unsigned int tot_itr = end_row * end_index;
	for (unsigned int itr = 0; itr < total_itr; itr++){
		#pragma HLS PIPELINE II=1
		#pragma HLS loop_tripcount min=min_grid max=max_grid avg=avg_grid
		t_pkt tmp;
		tmp.data = in.read();
		out.write(tmp);
	}
}


static void process_tile( hls::stream<uint256_dt> &rd_buffer, hls::stream<uint256_dt> &wr_buffer, struct data_G data_g){
	unsigned short xblocks = data_g.xblocks;
	unsigned short sizex = data_g.sizex;
	unsigned short sizey = data_g.sizey;
	unsigned short sizez = data_g.sizez;
	unsigned short limit_z = data_g.limit_z;
	unsigned short grid_sizey = data_g.grid_sizey;
	unsigned short grid_sizez = data_g.grid_sizez;
	unsigned short tile_y = data_g.tile_y;

	unsigned short offset_x = data_g.offset_x;
	unsigned short offset_y = data_g.offset_y;

	unsigned int line_diff = data_g.line_diff;
	unsigned int plane_diff = data_g.plane_diff;
	unsigned int gridsize = data_g.gridsize_pr;

	unsigned short batches = data_g.batches;
	unsigned int limit_read = data_g.gridsize_da;

	float s_1_1_2_arr[PORT_WIDTH];
	float s_1_2_1_arr[PORT_WIDTH];
	float s_1_1_1_arr[PORT_WIDTH+2];
	float s_1_0_1_arr[PORT_WIDTH];
	float s_1_1_0_arr[PORT_WIDTH];

	float mem_wr[PORT_WIDTH];

	#pragma HLS ARRAY_PARTITION variable=s_1_1_2_arr complete dim=1
	#pragma HLS ARRAY_PARTITION variable=s_1_2_1_arr complete dim=1
	#pragma HLS ARRAY_PARTITION variable=s_1_1_1_arr complete dim=1
	#pragma HLS ARRAY_PARTITION variable=s_1_0_1_arr complete dim=1
	#pragma HLS ARRAY_PARTITION variable=s_1_1_0_arr complete dim=1
	#pragma HLS ARRAY_PARTITION variable=mem_wr complete dim=1

	uint256_dt window_1[max_depth_xy];
	uint256_dt window_2[max_depth_8];
	uint256_dt window_3[max_depth_8];
	uint256_dt window_4[max_depth_xy];

	#pragma HLS RESOURCE variable=window_1 core=XPM_MEMORY uram latency=2
	#pragma HLS RESOURCE variable=window_2 core=RAM_1P_BRAM latency=2
	#pragma HLS RESOURCE variable=window_3 core=RAM_1P_BRAM latency=2
	#pragma HLS RESOURCE variable=window_4 core=XPM_MEMORY uram latency=2

	uint256_dt s_1_1_2, s_1_2_1, s_1_1_1, s_1_1_1_b, s_1_1_1_f, s_1_0_1, s_1_1_0;
	uint256_dt update_j;


	unsigned short i = 0, j = 0, k = 0;
	unsigned short j_p = 0, j_l = 0;
	for(unsigned int itr = 0; itr < gridsize; itr++) {
		#pragma HLS loop_tripcount min=min_grid max=max_grid avg=avg_grid
		#pragma HLS PIPELINE II=1
		bool cond_x = (k == xblocks);
		bool cond_y = (j == tile_y -1);
		bool cond_z = (i == limit_z - 1);

		if(k == xblocks){
			k = 0;
		}

		if(cond_y && cond_x){
			j = 0;
		}else if(cond_x){
			j++;
		}

		if(cond_x && cond_y && cond_z){
			i = 1;
		} else if(cond_y && cond_x){
			i++;
		}



		s_1_1_0 = window_4[j_p];

		s_1_0_1 = window_3[j_l];
		window_4[j_p] = s_1_0_1;

		s_1_1_1_b = s_1_1_1;
		window_3[j_l] = s_1_1_1_b;

		s_1_1_1 = s_1_1_1_f;
		s_1_1_1_f = window_2[j_l]; 	// read

		s_1_2_1 = window_1[j_p];   // read
		window_2[j_l] = s_1_2_1;	//set


		bool cond_tmp1 = (itr < limit_read);
		if(cond_tmp1){
			s_1_1_2 = rd_buffer.read(); // set
		}
		window_1[j_p] = s_1_1_2; // set



		j_p++;
		if(j_p == plane_diff){
			j_p = 0;
		}

		j_l++;
		if(j_l == line_diff){
			j_l = 0;
		}

		vec2arr: for(int k = 0; k < PORT_WIDTH; k++){
			#pragma HLS loop_tripcount min=port_width max=port_width avg=port_width
			data_conv s_1_1_2_u, s_1_2_1_u, s_1_1_1_u, s_1_0_1_u, s_1_1_0_u;
			s_1_1_2_u.i = s_1_1_2.range(DATATYPE_SIZE * (k + 1) - 1, k * DATATYPE_SIZE);
			s_1_2_1_u.i = s_1_2_1.range(DATATYPE_SIZE * (k + 1) - 1, k * DATATYPE_SIZE);
			s_1_1_1_u.i = s_1_1_1.range(DATATYPE_SIZE * (k + 1) - 1, k * DATATYPE_SIZE);
			s_1_0_1_u.i = s_1_0_1.range(DATATYPE_SIZE * (k + 1) - 1, k * DATATYPE_SIZE);
			s_1_1_0_u.i = s_1_1_0.range(DATATYPE_SIZE * (k + 1) - 1, k * DATATYPE_SIZE);

			s_1_1_2_arr[k]   =  s_1_1_2_u.f;
			s_1_2_1_arr[k]   =  s_1_2_1_u.f;
			s_1_1_1_arr[k+1] =  s_1_1_1_u.f;
			s_1_0_1_arr[k]   =  s_1_0_1_u.f;
			s_1_1_0_arr[k]   =  s_1_1_0_u.f;

		}
		data_conv tmp1_o1, tmp2_o2;
		tmp1_o1.i = s_1_1_1_b.range(DATATYPE_SIZE * (PORT_WIDTH) - 1, (PORT_WIDTH-1) * DATATYPE_SIZE);
		tmp2_o2.i = s_1_1_1_f.range(DATATYPE_SIZE * (0 + 1) - 1, 0 * DATATYPE_SIZE);
		s_1_1_1_arr[0] = tmp1_o1.f;
		s_1_1_1_arr[PORT_WIDTH + 1] = tmp2_o2.f;


		unsigned short y_index = j + offset_y;
		process: for(short q = 0; q < PORT_WIDTH; q++){
			#pragma HLS loop_tripcount min=port_width max=port_width avg=port_width
			short index = (k << SHIFT_BITS) + q + offset_x;
			float r1_1_2 =  s_1_1_2_arr[q] * (0.02f);
			float r1_2_1 =  s_1_2_1_arr[q] * (0.04f);
			float r0_1_1 =  s_1_1_1_arr[q] * (0.05f);
			float r1_1_1 =  s_1_1_1_arr[q+1] * (0.79f);
			float r2_1_1 =  s_1_1_1_arr[q+2] * (0.06f);
			float r1_0_1 =  s_1_0_1_arr[q] * (0.03f);
			float r1_1_0 =  s_1_1_0_arr[q] * (0.01f);

			float f1 = r1_1_2 + r1_2_1;
			float f2 = r0_1_1 + r1_1_1;
			float f3 = r2_1_1 + r1_0_1;

			#pragma HLS RESOURCE variable=f1 core=FAddSub_nodsp
			#pragma HLS RESOURCE variable=f2 core=FAddSub_nodsp
//			#pragma HLS RESOURCE variable=f3 core=FAddSub_nodsp

			float r1 = f1 + f2;
			float r2=  f3 + r1_1_0;

			float result  = r1 + r2;
			bool change_cond = register_it <bool>(index <= offset_x || index > sizex || (i <= 1) || (i >= limit_z -1) || (y_index <= 0) || (y_index >= grid_sizey -1));
			mem_wr[q] = change_cond ? s_1_1_1_arr[q+1] : result;
		}

		array2vec: for(int k = 0; k < PORT_WIDTH; k++){
			#pragma HLS loop_tripcount min=port_width max=port_width avg=port_width
			data_conv tmp;
			tmp.f = mem_wr[k];
			update_j.range(DATATYPE_SIZE * (k + 1) - 1, k * DATATYPE_SIZE) = tmp.i;
		}

		bool cond_wr = (i >= 1) && ( i < limit_z);
		if(cond_wr ) {
			wr_buffer << update_j;
		}

		// move the cell block
		k++;
	}
}
