template <int pidx>
static void derives_calc_ytep_k4( queue &q, struct data_G data_g,  ac_int<12,true> n_iter){
	
	event e1 = q.submit([&](handler &h) {
	// cl::sycl::stream out(1024, 256, h);
    h.single_task<class struct_idX<pidx>>([=] () [[intel::kernel_args_restrict]]{

		unsigned short grid_sizex = data_g.grid_sizex;
		unsigned short xblocks = data_g.xblocks;
		unsigned short sizex = data_g.sizex;
		unsigned short sizey = data_g.sizey;
		unsigned short sizez = data_g.sizez;
		unsigned short limit_z = data_g.limit_z;
		unsigned short grid_sizey = data_g.grid_sizey;
		unsigned short grid_sizez = data_g.grid_sizez;
		unsigned int line_diff = data_g.line_diff;
		unsigned int plane_diff = data_g.plane_diff;
		unsigned int plane_size = data_g.plane_size;
		unsigned int gridsize = data_g.gridsize_pr;

		[[intel::disable_loop_pipelining]]
		for(ac_int<12,true> u_itr = 0; u_itr < n_iter; u_itr++){

			struct dPath window_z_p_1[plane_buff_size];
			struct dPath window_z_p_2[plane_buff_size];
			struct dPath window_z_p_3[plane_buff_size];
			struct dPath window_z_p_4[plane_buff_size];

			struct dPath window_y_p_1[line_buff_size];
			struct dPath window_y_p_2[line_buff_size];
			struct dPath window_y_p_3[line_buff_size];
			struct dPath window_y_p_4[line_buff_size];

			struct dPath window_y_n_1[line_buff_size];
			struct dPath window_y_n_2[line_buff_size];
			struct dPath window_y_n_3[line_buff_size];
			struct dPath window_y_n_4[line_buff_size];

			struct dPath window_z_n_1[plane_buff_size];
			struct dPath window_z_n_2[plane_buff_size];
			struct dPath window_z_n_3[plane_buff_size];
			struct dPath window_z_n_4[plane_buff_size];

			struct dPath16 window_yy_16[4*plane_buff_size];
			struct dPath2 window_yy_2[4*plane_buff_size];
			struct dPath16 window_yy_final_16[4*plane_buff_size];
			struct dPath2 window_yy_final_2[4*plane_buff_size];


			struct dPath s_4_4_8, s_4_4_7, s_4_4_6, s_4_4_5;
			struct dPath s_4_8_4, s_4_7_4, s_4_6_4, s_4_5_4;
			struct dPath s_8_4_4, s_7_4_4, s_6_4_4, s_5_4_4;
			struct dPath s_4_4_4;
			struct dPath s_3_4_4, s_2_4_4, s_1_4_4, s_0_4_4;
			struct dPath s_4_3_4, s_4_2_4, s_4_1_4, s_4_0_4;
			struct dPath s_4_4_3, s_4_4_2, s_4_4_1, s_4_4_0;
			struct dPath yy_vec_tmp;
			struct dPath yy_final_vec_tmp, yy_final_vec_write;
			struct dPath update_j;


			const float c[2*ORDER+1] = {0.0035714285714285713,-0.0380952380952381,0.2,-0.8,0.0,0.8,-0.2,0.0380952380952381,-0.0035714285714285713};
			const float invdx = 200; // 1/0.005
			const float invdy = 200; // 1/0.005
			const float invdz = 200; // 1/0.005
			const unsigned short half = 4;
			const unsigned short pml_width = 10;

			short xbeg=half;
			short xend=sizex-half;
			short ybeg=half;
			short yend=sizey-half;
			short zbeg=half;
			short zend=sizez-half;
			short xpmlbeg=xbeg+pml_width;
			short ypmlbeg=ybeg+pml_width;
			short zpmlbeg=zbeg+pml_width;
			short xpmlend=xend-pml_width;
			short ypmlend=yend-pml_width;
			short zpmlend=zend-pml_width;



			
			unsigned short i = 0 , j = 0, k = 0;
			unsigned short i_dum = 0, j_dum = 0, k_dum = 0;
			unsigned short j_p_dum = 0, j_l_dum = 0, j_p_diff_dum = 0, j_l_diff_dum = 0, j_p_4_dum = 0;
			unsigned short j_p, j_l, j_p_diff, j_l_diff, j_p_4;

			// out << "Itr: " << gridsize << "\n";
			[[intel::initiation_interval(1)]]
			for(unsigned int itr = 0; itr < gridsize; itr++) {
				// printf("itr:%d\n", itr);

				#pragma HLS loop_tripcount min=min_grid max=max_grid avg=avg_grid
				#pragma HLS PIPELINE II=1

				j_p = j_p_dum;
				j_l = j_l_dum;
				j_p_diff = j_p_diff_dum;
				j_l_diff = j_l_diff_dum;
				j_p_4 = j_p_4_dum;


				bool cmp1 = (k == xblocks -1);
				bool cmp2 = cmp1 && (j == grid_sizey-1);
				bool cmp3 = cmp2 && (i == limit_z-1);

				if(cmp1){
					k_dum = 0;
				}

				if(cmp2){
					j_dum = 0;
				} else if(cmp1){
					j_dum = j + 1;
				}

				if(cmp3){
					i_dum = 0;
				} else if(cmp2){
					i_dum = i + 1;
				}

				i = i_dum;
				j = j_dum;
				k = k_dum;

				// negetive z arm
				s_4_4_0 = window_z_n_4[j_p];

				s_4_4_1 = window_z_n_3[j_p];
				window_z_n_4[j_p] = s_4_4_1;

				s_4_4_2 = window_z_n_2[j_p];
				window_z_n_3[j_p] = s_4_4_2;

				s_4_4_3 = window_z_n_1[j_p_diff];
				window_z_n_2[j_p] = s_4_4_3;


				// Negetive y arm
				s_4_0_4 = window_y_n_4[j_l]; 
				window_z_n_1[j_p_diff] = s_4_0_4;

				s_4_1_4 = window_y_n_3[j_l];
				window_y_n_4[j_l] = s_4_1_4;

				s_4_2_4 = window_y_n_2[j_l];
				window_y_n_3[j_l] = s_4_2_4;

				s_4_3_4 = window_y_n_1[j_l_diff];
				window_y_n_2[j_l] = s_4_3_4;


				// negetive to positive x arm
				// here you have reduce the bus width

				s_2_4_4 = s_3_4_4;
				window_y_n_1[j_l_diff] = s_2_4_4;
				s_3_4_4 = s_4_4_4;
				s_4_4_4 = s_5_4_4;
				s_5_4_4 = s_6_4_4;


				// positive Y arm 
				s_6_4_4 = window_y_p_1[j_l_diff];



				s_4_5_4 = window_y_p_2[j_l];
				window_y_p_1[j_l_diff] = s_4_5_4;

				s_4_6_4 = window_y_p_3[j_l];
				window_y_p_2[j_l] = s_4_6_4;

				s_4_7_4 = window_y_p_4[j_l];
				window_y_p_3[j_l] = s_4_7_4;



				// positive z arm
				s_4_8_4 = window_z_p_1[j_p_diff];
				window_y_p_4[j_l] = s_4_8_4;

				s_4_4_5 = window_z_p_2[j_p];
				window_z_p_1[j_p_diff] = s_4_4_5;

				s_4_4_6 = window_z_p_3[j_p];
				window_z_p_2[j_p] = s_4_4_6; 	

				s_4_4_7 = window_z_p_4[j_p];   //set
				window_z_p_3[j_p] = s_4_4_7;   //set	

				bool cond_tmp1 = (i < grid_sizez);
				if(cond_tmp1){
					s_4_4_8 = pipeM::PipeAt<pidx>::read();  // set
				}
				window_z_p_4[j_p] = s_4_4_8; // set



				struct dPath16 yy_vec16 = window_yy_16[j_p_4];
				struct dPath2 yy_vec2 = window_yy_2[j_p_4];
				struct dPath yy_vec = {{s_4_4_4.data[0], s_4_4_4.data[1], yy_vec2.data[0], yy_vec2.data[1], yy_vec16.data[0], yy_vec16.data[1], yy_vec16.data[2], yy_vec16.data[3], \
										s_4_4_4.data[8], s_4_4_4.data[9], yy_vec16.data[4], yy_vec16.data[5], yy_vec16.data[6], yy_vec16.data[7], yy_vec16.data[8], yy_vec16.data[9], \
										s_4_4_4.data[16], s_4_4_4.data[17], yy_vec16.data[10], yy_vec16.data[11], yy_vec16.data[12], yy_vec16.data[13], yy_vec16.data[14], yy_vec16.data[15]}};


				bool cond_tmp2 = (i < grid_sizez);
				if(cond_tmp1){
					yy_vec_tmp = pipeM::PipeAt<pidx+10>::read(); // set
				}

				struct dPath2 yy_vec_tmp2 = {{yy_vec_tmp.data[2], yy_vec_tmp.data[3]}};
				struct dPath16 yy_vec_tmp16 = {{yy_vec_tmp.data[4], yy_vec_tmp.data[5], yy_vec_tmp.data[6], yy_vec_tmp.data[7], \
												yy_vec_tmp.data[10], yy_vec_tmp.data[11], yy_vec_tmp.data[12], yy_vec_tmp.data[13], yy_vec_tmp.data[14], yy_vec_tmp.data[15], \
												yy_vec_tmp.data[18], yy_vec_tmp.data[19], yy_vec_tmp.data[20], yy_vec_tmp.data[21], yy_vec_tmp.data[22], yy_vec_tmp.data[23]}};

				window_yy_2[j_p_4] = yy_vec_tmp2;
				window_yy_16[j_p_4] = yy_vec_tmp16;






				struct dPath2 yy_final_vec2 = window_yy_final_2[j_p_4];
				struct dPath16 yy_final_vec16 = window_yy_final_16[j_p_4];
				struct dPath yy_final_vec = {{s_4_4_4.data[0], s_4_4_4.data[1], yy_final_vec2.data[0], yy_final_vec2.data[1], yy_final_vec16.data[0], yy_final_vec16.data[1], yy_final_vec16.data[2], yy_final_vec16.data[3], \
											  s_4_4_4.data[8], s_4_4_4.data[9], yy_final_vec16.data[4], yy_final_vec16.data[5], yy_final_vec16.data[6], yy_final_vec16.data[7], yy_final_vec16.data[8], yy_final_vec16.data[9], \
											  s_4_4_4.data[16], s_4_4_4.data[17], yy_final_vec16.data[10], yy_final_vec16.data[11], yy_final_vec16.data[12], yy_final_vec16.data[13], yy_final_vec16.data[14], yy_final_vec16.data[15]}};
				if(cond_tmp1){
					yy_final_vec_tmp = pipeM::PipeAt<pidx+20>::read(); // set
				}

				struct dPath2 yy_final_vec_tmp2 = {{yy_final_vec_tmp.data[2], yy_final_vec_tmp.data[3]}};
				struct dPath16 yy_final_vec_tmp16 = {{yy_final_vec_tmp.data[4], yy_final_vec_tmp.data[5], yy_final_vec_tmp.data[6], yy_final_vec_tmp.data[7], \
													  yy_final_vec_tmp.data[10], yy_final_vec_tmp.data[11], yy_final_vec_tmp.data[12], yy_final_vec_tmp.data[13], yy_final_vec_tmp.data[14], yy_final_vec_tmp.data[15], \
													  yy_final_vec_tmp.data[18], yy_final_vec_tmp.data[19], yy_final_vec_tmp.data[20], yy_final_vec_tmp.data[21], yy_final_vec_tmp.data[22], yy_final_vec_tmp.data[23]}};

				window_yy_final_2[j_p_4] = yy_final_vec_tmp2;
				window_yy_final_16[j_p_4] = yy_final_vec_tmp16;



				if(j_p >= plane_size-1){
					j_p_dum = 0;
				} else {
					j_p_dum++;
				}

				
				if(j_l >= xblocks-1){
					j_l_dum = 0;
				} else {
					j_l_dum++;
				}

				
				if(j_p_diff >= plane_diff - 1){
					j_p_diff_dum = 0;
				} else {
					j_p_diff_dum++;
				}

				
				if(j_l_diff >= line_diff - 1){
					j_l_diff_dum = 0;
				} else {
					j_l_diff_dum++;
				}

				
				if(j_p_4 >= (plane_size << 2) -1){
					j_p_4_dum = 0;
				} else {
					j_p_4_dum++;
				}



				// X ARM
				#define x_ARM(x)   {s_2_4_4.data[INC2((x))], s_3_4_4.data[INC0((x))], s_3_4_4.data[INC1((x))], s_3_4_4.data[INC2((x))], s_4_4_4.data[INC0((x))], s_4_4_4.data[INC1((x))], s_4_4_4.data[INC2((x))], s_5_4_4.data[INC0((x))], s_5_4_4.data[INC1((x))], \
									s_3_4_4.data[INC0((x))], s_3_4_4.data[INC1((x))], s_3_4_4.data[INC2((x))], s_4_4_4.data[INC0((x))], s_4_4_4.data[INC1((x))], s_4_4_4.data[INC2((x))], s_5_4_4.data[INC0((x))], s_5_4_4.data[INC1((x))], s_5_4_4.data[INC2((x))], \
									s_3_4_4.data[INC1((x))], s_3_4_4.data[INC2((x))], s_4_4_4.data[INC0((x))], s_4_4_4.data[INC1((x))], s_4_4_4.data[INC2((x))], s_5_4_4.data[INC0((x))], s_5_4_4.data[INC1((x))], s_5_4_4.data[INC2((x))], s_6_4_4.data[INC0((x))]} 





				float X_ARM_0[(2*ORDER+1)*3] = x_ARM(2);
				float X_ARM_1[(2*ORDER+1)*3] = x_ARM(3);
				float X_ARM_2[(2*ORDER+1)*3] = x_ARM(4);
				float X_ARM_3[(2*ORDER+1)*3] = x_ARM(5);
				float X_ARM_4[(2*ORDER+1)*3] = x_ARM(6);
				float X_ARM_5[(2*ORDER+1)*3] = x_ARM(7);

				#undef x_ARM(x)


			
				// Y ARM
				#define y_ARM(x) {s_4_0_4.data[INC0((x))], s_4_1_4.data[INC0((x))], s_4_2_4.data[INC0((x))], s_4_3_4.data[INC0((x))], s_4_4_4.data[INC0((x))], s_4_5_4.data[INC0((x))], s_4_6_4.data[INC0((x))], s_4_7_4.data[INC0((x))], s_4_8_4.data[INC0((x))], \
								  s_4_0_4.data[INC1((x))], s_4_1_4.data[INC1((x))], s_4_2_4.data[INC1((x))], s_4_3_4.data[INC1((x))], s_4_4_4.data[INC1((x))], s_4_5_4.data[INC1((x))], s_4_6_4.data[INC1((x))], s_4_7_4.data[INC1((x))], s_4_8_4.data[INC1((x))], \
								  s_4_0_4.data[INC2((x))], s_4_1_4.data[INC2((x))], s_4_2_4.data[INC2((x))], s_4_3_4.data[INC2((x))], s_4_4_4.data[INC2((x))], s_4_5_4.data[INC2((x))], s_4_6_4.data[INC2((x))], s_4_7_4.data[INC2((x))], s_4_8_4.data[INC2((x))]}


				float Y_ARM_0[(2*ORDER+1)*3] = y_ARM(2); 
				float Y_ARM_1[(2*ORDER+1)*3] = y_ARM(3); 
				float Y_ARM_2[(2*ORDER+1)*3] = y_ARM(4); 
				float Y_ARM_3[(2*ORDER+1)*3] = y_ARM(5); 
				float Y_ARM_4[(2*ORDER+1)*3] = y_ARM(6); 
				float Y_ARM_5[(2*ORDER+1)*3] = y_ARM(7); 

				#undef y_ARM(x)
				

				// Z ARM

				#define z_ARM(x) {s_4_4_0.data[INC0((x))], s_4_4_1.data[INC0((x))], s_4_4_2.data[INC0((x))], s_4_4_3.data[INC0((x))], s_4_4_4.data[INC0((x))], s_4_4_5.data[INC0((x))], s_4_4_6.data[INC0((x))], s_4_4_7.data[INC0((x))], s_4_4_8.data[INC0((x))], \
								  s_4_4_0.data[INC1((x))], s_4_4_1.data[INC1((x))], s_4_4_2.data[INC1((x))], s_4_4_3.data[INC1((x))], s_4_4_4.data[INC1((x))], s_4_4_5.data[INC1((x))], s_4_4_6.data[INC1((x))], s_4_4_7.data[INC1((x))], s_4_4_8.data[INC1((x))], \
								  s_4_4_0.data[INC2((x))], s_4_4_1.data[INC2((x))], s_4_4_2.data[INC2((x))], s_4_4_3.data[INC2((x))], s_4_4_4.data[INC2((x))], s_4_4_5.data[INC2((x))], s_4_4_6.data[INC2((x))], s_4_4_7.data[INC2((x))], s_4_4_8.data[INC2((x))]}

				float Z_ARM_0[(2*ORDER+1)*3] = z_ARM(2);
				float Z_ARM_1[(2*ORDER+1)*3] = z_ARM(3);
				float Z_ARM_2[(2*ORDER+1)*3] = z_ARM(4);
				float Z_ARM_3[(2*ORDER+1)*3] = z_ARM(5);
				float Z_ARM_4[(2*ORDER+1)*3] = z_ARM(6);
				float Z_ARM_5[(2*ORDER+1)*3] = z_ARM(7);

				#undef z_ARM(x)



				struct dPath wr_vec;
				struct dPath wr_vec_dt;
				struct dPath y_tmp_vec;
				struct dPath y_final_vec;
				struct dPath yyTmp_vec;

				#pragma unroll 3
				for(int v = 0; v < 3; v++){

				  	float sigma = s_4_4_4.data[1+v*8]/s_4_4_4.data[0+v*8]; //mu[OPS_ACC5(0,0,0)]/rho[OPS_ACC4(0,0,0)];
				  	float sigmax=0.0f;
				  	float sigmay=0.0f;
				  	float sigmaz=0.0f;

				  	short idx = k*3 - ORDER;
				  	short idy = j - ORDER;
				  	short idz = i - 2*ORDER;


				  	float sigmax1, sigmax2, sigmay1, sigmay2, sigmaz1, sigmaz2;


				  	bool idx_cond1 = idx <= xbeg+pml_width;
				  	bool idx_cond2 = idx >=xend-pml_width;
				  	sigmax1 = (xbeg+pml_width-idx ) * sigma * 0.1f;//sigma/pml_width;
				  	sigmax2 =(idx -(xend-pml_width)) * sigma * 0.1f; //sigma/pml_width;
				  	sigmax = idx_cond2 ? sigmax2 : (idx_cond1 ? sigmax1 : 0.0f);

				  	bool idy_cond1 = idy <= ybeg+pml_width;
				  	bool idy_cond2 = idy >= yend-pml_width;
				  	sigmay1=(ybeg+pml_width-idy) * sigma * 0.1f; //sigma/pml_width;
				  	sigmay2=(idy-(yend-pml_width)) * sigma * 0.1f; //sigma/pml_width;
				  	sigmay = idy_cond2 ? sigmay2 : (idy_cond1 ? sigmay1 : 0.0f);

				  	bool idz_cond1 = idz <= zbeg+pml_width;
				  	bool idz_cond2 = idz >= zend-pml_width;
				  	sigmaz1=(zbeg+pml_width-idz) * sigma * 0.1f; //sigma/pml_width;
				  	sigmaz2=(idz -(zend-pml_width)) * sigma * 0.1f; //sigma/pml_width;
				  	sigmaz = idz_cond2 ? sigmaz2 : (idz_cond1 ? sigmaz1 : 0.0f);


				  	float px = X_ARM_0[4+v*9];
				  	float py = X_ARM_1[4+v*9];
				  	float pz = X_ARM_2[4+v*9];

				  	float vx = X_ARM_3[4+v*9];
				  	float vy = X_ARM_4[4+v*9];
				  	float vz = X_ARM_5[4+v*9];


				  	float vxx=0.0;
				  	float vxy=0.0;
				  	float vxz=0.0;
				  	
				  	float vyx=0.0;
				  	float vyy=0.0;
				  	float vyz=0.0;
				
				  	float vzx=0.0;
				  	float vzy=0.0;
				  	float vzz=0.0;
				  	
				  	float pxx=0.0;
				  	float pxy=0.0;
				  	float pxz=0.0;
				  	
				  	float pyx=0.0;
				  	float pyy=0.0;
				  	float pyz=0.0;
				
				  	float pzx=0.0;
				  	float pzy=0.0;
				  	float pzz=0.0;

				  	const int iU_FACTOR = ORDER*2+1;
				  	#pragma unroll iU_FACTOR
				  	for(int l=0;l <= ORDER*2; l++){
					    pxx += X_ARM_0[l+v*9] * c[l];
					    pyx += X_ARM_1[l+v*9] * c[l];
					    pzx += X_ARM_2[l+v*9] * c[l];
					    
					    vxx += X_ARM_3[l+v*9] * c[l];
					    vyx += X_ARM_4[l+v*9] * c[l];
					    vzx += X_ARM_5[l+v*9] * c[l];
					    
					    pxy += Y_ARM_0[l+v*9] * c[l];
					    pyy += Y_ARM_1[l+v*9] * c[l];
					    pzy += Y_ARM_2[l+v*9] * c[l];
					    
					    vxy += Y_ARM_3[l+v*9] * c[l];
					    vyy += Y_ARM_4[l+v*9] * c[l];
					    vzy += Y_ARM_5[l+v*9] * c[l];
					    
					    pxz += Z_ARM_0[l+v*9] * c[l];
					    pyz += Z_ARM_1[l+v*9] * c[l];
					    pzz += Z_ARM_2[l+v*9] * c[l];
					    
					    vxz += Z_ARM_3[l+v*9] * c[l];
					    vyz += Z_ARM_4[l+v*9] * c[l];
					    vzz += Z_ARM_5[l+v*9] * c[l];

					}

				  	pxx *= invdx;
				  	pyx *= invdx;
					pzx *= invdx;

					vxx *= invdx;
					vyx *= invdx;
					vzx *= invdx;

					pxy *= invdy;
					pyy *= invdy;
					pzy *= invdy;

					vxy *= invdy;
					vyy *= invdy;
					vzy *= invdy;

					pxz *= invdz;
					pyz *= invdz;
					pzz *= invdz;

					vxz *= invdz;
					vyz *= invdz;
					vzz *= invdz;


					struct dPath wr_vec;

					float k_2_1 = vxx/s_4_4_4.data[0+v*8];
					float k_2_2 =  sigmax*px;
					wr_vec.data[2+v*8]= k_2_1 - k_2_2;            //vxx/rho[OPS_ACC4(0,0,0)] - sigmax*px;

					float k_5_1 = (pxx+pyx+pxz);
					float k_5_2 = k_5_1 *s_4_4_4.data[1+v*8];
					float k_5_3 = sigmax*vx;
					wr_vec.data[5+v*8]= k_5_2 - k_5_3;  //(pxx+pyx+pxz)*mu[OPS_ACC5(0,0,0)] - sigmax*vx;

					float k_3_1 = vyy/s_4_4_4.data[0+v*8];
					float k_3_2 = sigmay*py;
					wr_vec.data[3+v*8]= k_3_1 - k_3_2;  		  // vyy/rho[OPS_ACC4(0,0,0)] - sigmay*py;

					float k_6_1 = (pxy+pyy+pyz);
					float k_6_2 = k_6_1 *s_4_4_4.data[1+v*8];
					float k_6_3 = sigmay*vy;
					wr_vec.data[6+v*8]= k_6_2 - k_6_3;   //(pxy+pyy+pyz)*mu[OPS_ACC5(0,0,0)] - sigmay*vy;

					float k_4_1 = vzz/s_4_4_4.data[0+v*8];
					float k_4_2 = sigmaz*pz;
					wr_vec.data[4+v*8]= k_4_1  - k_4_2;  		  //vzz/rho[OPS_ACC4(0,0,0)] - sigmaz*pz;

					float k_7_1 = (pxz+pyz+pzz);
					float k_7_2 = k_7_1*s_4_4_4.data[1+v*8];
					float k_7_3 = sigmaz*vz;
					wr_vec.data[7+v*8]= k_7_2 - k_7_3;  //(pxz+pyz+pzz)*mu[OPS_ACC5(0,0,0)] - sigmaz*vz;

			  		wr_vec.data[0+v*8] = s_4_4_4.data[0+v*8];
			  		wr_vec.data[1+v*8] = s_4_4_4.data[1+v*8];


			  		// calc K dt
			  		struct dPath wr_vec_dt;
			  		wr_vec_dt.data[2+v*8] = wr_vec.data[2+v*8] * 0.1f;
			  		wr_vec_dt.data[5+v*8] = wr_vec.data[5+v*8] * 0.1f;
			  		
			  		wr_vec_dt.data[3+v*8] = wr_vec.data[3+v*8] * 0.1f;
			  		wr_vec_dt.data[6+v*8] = wr_vec.data[6+v*8] * 0.1f;
			  		
			  		wr_vec_dt.data[4+v*8] = wr_vec.data[4+v*8] * 0.1f;
			  		wr_vec_dt.data[7+v*8] = wr_vec.data[7+v*8] * 0.1f;

			  		wr_vec_dt.data[0+v*8] = s_4_4_4.data[0+v*8];
			  		wr_vec_dt.data[1+v*8] = s_4_4_4.data[1+v*8];


			  		// YY final 
			  		yy_final_vec.data[2+v*8] += wr_vec_dt.data[2+v*8] * 0.1666666667f;
			  		yy_final_vec.data[5+v*8] += wr_vec_dt.data[5+v*8] * 0.1666666667f;

			  		yy_final_vec.data[3+v*8] += wr_vec_dt.data[3+v*8] * 0.1666666667f;
			  		yy_final_vec.data[6+v*8] += wr_vec_dt.data[6+v*8] * 0.1666666667f;

			  		yy_final_vec.data[4+v*8] += wr_vec_dt.data[4+v*8] * 0.1666666667f;
			  		yy_final_vec.data[7+v*8] += wr_vec_dt.data[7+v*8] * 0.1666666667f;

			  		yy_final_vec.data[0+v*8] = wr_vec_dt.data[0+v*8];
			  		yy_final_vec.data[1+v*8] = wr_vec_dt.data[1+v*8];


			  		// calc Y temp
			  		struct dPath yyTmp_vec;
			  		yyTmp_vec.data[2+v*8] = yy_vec.data[2+v*8] + wr_vec_dt.data[2+v*8];
			  		yyTmp_vec.data[5+v*8] = yy_vec.data[5+v*8] + wr_vec_dt.data[5+v*8];
			  	
			  		yyTmp_vec.data[3+v*8] = yy_vec.data[3+v*8] + wr_vec_dt.data[3+v*8];
			  		yyTmp_vec.data[6+v*8] = yy_vec.data[6+v*8] + wr_vec_dt.data[6+v*8];
			  	
			  		yyTmp_vec.data[4+v*8] = yy_vec.data[4+v*8] + wr_vec_dt.data[4+v*8];
			  		yyTmp_vec.data[7+v*8] = yy_vec.data[7+v*8] + wr_vec_dt.data[7+v*8];

			  		yyTmp_vec.data[0+v*8] = s_4_4_4.data[0+v*8];
			  		yyTmp_vec.data[1+v*8] = s_4_4_4.data[1+v*8];

			  		bool change_cond1 = (idx < 0) || (idx >= sizex) || (idy < 0) ;
			  		bool change_cond2 = (idy >= sizey ) || (idz < 0) || (idz >= sizez);
			  		bool change_cond = change_cond1 || change_cond2;

			  		#pragma unroll 8
			  		for(int ptr = v*8; ptr < (v+1)*8; ptr++){
						update_j.data[ptr] = change_cond ? s_4_4_4.data[ptr] : wr_vec.data[ptr];
						yy_final_vec_write.data[ptr] = change_cond ? s_4_4_4.data[ptr] : yy_final_vec.data[ptr];
					}
				}

				bool cond_wr = (i >= ORDER) && ( i < grid_sizez + ORDER);
				if(cond_wr ) {
					pipeM::PipeAt<pidx+1>::write(yy_final_vec_write); 
				}

				// move the cell block
				k_dum = k + 1;
			}
		}
	});
    });
}
