static void derives_calc_ytep_k3( queue &q, struct data_G data_g){
	
	event e1 = q.submit([&](handler &h) {
    h.single_task<class derives_calc_ytep_k3>([=] () [[intel::kernel_args_restrict]]{

		unsigned short grid_sizex = data_g.grid_sizex;
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

		struct dPath window_yy[4*plane_buff_size];
		struct dPath window_yy_final[4*plane_buff_size];



		struct dPath s_4_4_8, s_4_4_7, s_4_4_6, s_4_4_5;
		struct dPath s_4_8_4, s_4_7_4, s_4_6_4, s_4_5_4;
		struct dPath s_8_4_4, s_7_4_4, s_6_4_4, s_5_4_4;
		struct dPath s_4_4_4;
		struct dPath s_3_4_4, s_2_4_4, s_1_4_4, s_0_4_4;
		struct dPath s_4_3_4, s_4_2_4, s_4_1_4, s_4_0_4;
		struct dPath s_4_4_3, s_4_4_2, s_4_4_1, s_4_4_0;
		struct dPath yy_vec, yy_vec_tmp;
		struct dPath yy_final_vec, yy_final_vec_tmp, yy_final_vec_write;
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



		
		unsigned short i = 0, j = 0, k = 0;
		unsigned short i_dum = 0, j_dum = 0, k_dum = 0;
		unsigned short j_p_dum = 0, j_l_dum = 0, j_p_diff_dum = 0, j_l_diff_dum = 0, j_p_4_dum = 0;
		unsigned short j_p, j_l, j_p_diff, j_l_diff, j_p_4;
		for(unsigned int itr = 0; itr < gridsize; itr++) {
			#pragma HLS loop_tripcount min=min_grid max=max_grid avg=avg_grid
			#pragma HLS PIPELINE II=1

			j_p = j_p_dum;
			j_l = j_l_dum;
			j_p_diff = j_p_diff_dum;
			j_l_diff = j_l_diff_dum;
			j_p_4 = j_p_4_dum;


			bool cmp1 = (k == grid_sizex -1);
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


			s_0_4_4 = s_1_4_4;
			window_y_n_1[j_l_diff] = s_0_4_4;

			s_1_4_4 = s_2_4_4;
			s_2_4_4 = s_3_4_4;
			s_3_4_4 = s_4_4_4;
			s_4_4_4 = s_5_4_4;
			s_5_4_4 = s_6_4_4;
			s_6_4_4 = s_7_4_4;
			s_7_4_4 = s_8_4_4;


			// positive Y arm 
			s_8_4_4 = window_y_p_1[j_l_diff];



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
				s_4_4_8 = pipeS::PipeAt<2>::read(); // set
			}
			window_z_p_4[j_p] = s_4_4_8; // set



			yy_vec = window_yy[j_p_4];
			bool cond_tmp2 = (i < grid_sizez);
			if(cond_tmp1){
				yy_vec_tmp = pipeS::PipeAt<12>::read(); // set
			}
			window_yy[j_p_4] = yy_vec_tmp;


			yy_final_vec = window_yy_final[j_p_4];
			if(cond_tmp1){
				yy_final_vec_tmp = pipeS::PipeAt<22>::read(); // set
			}
			window_yy_final[j_p_4] = yy_final_vec_tmp;



			j_p_dum++;
			if(j_p_dum >= plane_size){
				j_p_dum = 0;
			}

			j_l_dum++;
			if(j_l_dum >= grid_sizex){
				j_l_dum = 0;
			}

			j_p_diff_dum++;
			if(j_p_diff_dum >= plane_diff){
				j_p_diff_dum = 0;
			}

			j_l_diff_dum++;
			if(j_l_diff_dum >= line_diff){
				j_l_diff_dum = 0;
			}

			j_p_4_dum++;
			if(j_p_4_dum >= (plane_size << 2)){
				j_p_4_dum = 0;
			}


			// X ARM

			float X_ARM_0[2*ORDER+1] = {s_0_4_4.data[2], s_1_4_4.data[2], s_2_4_4.data[2], s_3_4_4.data[2], s_4_4_4.data[2], s_5_4_4.data[2], s_6_4_4.data[2], s_7_4_4.data[2], s_8_4_4.data[2]}; 
			float X_ARM_1[2*ORDER+1] = {s_0_4_4.data[3], s_1_4_4.data[3], s_2_4_4.data[3], s_3_4_4.data[3], s_4_4_4.data[3], s_5_4_4.data[3], s_6_4_4.data[3], s_7_4_4.data[3], s_8_4_4.data[3]}; 
			float X_ARM_2[2*ORDER+1] = {s_0_4_4.data[4], s_1_4_4.data[4], s_2_4_4.data[4], s_3_4_4.data[4], s_4_4_4.data[4], s_5_4_4.data[4], s_6_4_4.data[4], s_7_4_4.data[4], s_8_4_4.data[4]}; 
			float X_ARM_3[2*ORDER+1] = {s_0_4_4.data[5], s_1_4_4.data[5], s_2_4_4.data[5], s_3_4_4.data[5], s_4_4_4.data[5], s_5_4_4.data[5], s_6_4_4.data[5], s_7_4_4.data[5], s_8_4_4.data[5]}; 
			float X_ARM_4[2*ORDER+1] = {s_0_4_4.data[6], s_1_4_4.data[6], s_2_4_4.data[6], s_3_4_4.data[6], s_4_4_4.data[6], s_5_4_4.data[6], s_6_4_4.data[6], s_7_4_4.data[6], s_8_4_4.data[6]}; 
			float X_ARM_5[2*ORDER+1] = {s_0_4_4.data[7], s_1_4_4.data[7], s_2_4_4.data[7], s_3_4_4.data[7], s_4_4_4.data[7], s_5_4_4.data[7], s_6_4_4.data[7], s_7_4_4.data[7], s_8_4_4.data[7]};

		
			// Y ARM

			float Y_ARM_0[2*ORDER+1] = {s_4_0_4.data[2], s_4_1_4.data[2], s_4_2_4.data[2], s_4_3_4.data[2], s_4_4_4.data[2], s_4_5_4.data[2], s_4_6_4.data[2], s_4_7_4.data[2], s_4_8_4.data[2]}; 
			float Y_ARM_1[2*ORDER+1] = {s_4_0_4.data[3], s_4_1_4.data[3], s_4_2_4.data[3], s_4_3_4.data[3], s_4_4_4.data[3], s_4_5_4.data[3], s_4_6_4.data[3], s_4_7_4.data[3], s_4_8_4.data[3]}; 
			float Y_ARM_2[2*ORDER+1] = {s_4_0_4.data[4], s_4_1_4.data[4], s_4_2_4.data[4], s_4_3_4.data[4], s_4_4_4.data[4], s_4_5_4.data[4], s_4_6_4.data[4], s_4_7_4.data[4], s_4_8_4.data[4]}; 
			float Y_ARM_3[2*ORDER+1] = {s_4_0_4.data[5], s_4_1_4.data[5], s_4_2_4.data[5], s_4_3_4.data[5], s_4_4_4.data[5], s_4_5_4.data[5], s_4_6_4.data[5], s_4_7_4.data[5], s_4_8_4.data[5]};
			float Y_ARM_4[2*ORDER+1] = {s_4_0_4.data[6], s_4_1_4.data[6], s_4_2_4.data[6], s_4_3_4.data[6], s_4_4_4.data[6], s_4_5_4.data[6], s_4_6_4.data[6], s_4_7_4.data[6], s_4_8_4.data[6]};
			float Y_ARM_5[2*ORDER+1] = {s_4_0_4.data[7], s_4_1_4.data[7], s_4_2_4.data[7], s_4_3_4.data[7], s_4_4_4.data[7], s_4_5_4.data[7], s_4_6_4.data[7], s_4_7_4.data[7], s_4_8_4.data[7]};
			

			// Z ARM

			float Z_ARM_0[2*ORDER+1] = {s_4_4_0.data[2], s_4_4_1.data[2], s_4_4_2.data[2], s_4_4_3.data[2], s_4_4_4.data[2], s_4_4_5.data[2], s_4_4_6.data[2], s_4_4_7.data[2], s_4_4_8.data[2]}; 
			float Z_ARM_1[2*ORDER+1] = {s_4_4_0.data[3], s_4_4_1.data[3], s_4_4_2.data[3], s_4_4_3.data[3], s_4_4_4.data[3], s_4_4_5.data[3], s_4_4_6.data[3], s_4_4_7.data[3], s_4_4_8.data[3]}; 
			float Z_ARM_2[2*ORDER+1] = {s_4_4_0.data[4], s_4_4_1.data[4], s_4_4_2.data[4], s_4_4_3.data[4], s_4_4_4.data[4], s_4_4_5.data[4], s_4_4_6.data[4], s_4_4_7.data[4], s_4_4_8.data[4]}; 
			float Z_ARM_3[2*ORDER+1] = {s_4_4_0.data[5], s_4_4_1.data[5], s_4_4_2.data[5], s_4_4_3.data[5], s_4_4_4.data[5], s_4_4_5.data[5], s_4_4_6.data[5], s_4_4_7.data[5], s_4_4_8.data[5]};
			float Z_ARM_4[2*ORDER+1] = {s_4_4_0.data[6], s_4_4_1.data[6], s_4_4_2.data[6], s_4_4_3.data[6], s_4_4_4.data[6], s_4_4_5.data[6], s_4_4_6.data[6], s_4_4_7.data[6], s_4_4_8.data[6]};
			float Z_ARM_5[2*ORDER+1] = {s_4_4_0.data[7], s_4_4_1.data[7], s_4_4_2.data[7], s_4_4_3.data[7], s_4_4_4.data[7], s_4_4_5.data[7], s_4_4_6.data[7], s_4_4_7.data[7], s_4_4_8.data[7]};


		  	float sigma = s_4_4_4.data[1]/s_4_4_4.data[0]; //mu[OPS_ACC5(0,0,0)]/rho[OPS_ACC4(0,0,0)];
		  	float sigmax=0.0;
		  	float sigmay=0.0;
		  	float sigmaz=0.0;

		  	short idx = k - ORDER;
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


		  	float px = X_ARM_0[4];
		  	float py = X_ARM_1[4];
		  	float pz = X_ARM_2[4];

		  	float vx = X_ARM_3[4];
		  	float vy = X_ARM_4[4];
		  	float vz = X_ARM_5[4];


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

		  	for(int l=0;l <= ORDER*2; l++){
			    pxx += X_ARM_0[l] * c[l];
			    pyx += X_ARM_1[l] * c[l];
			    pzx += X_ARM_2[l] * c[l];
			    
			    vxx += X_ARM_3[l] * c[l];
			    vyx += X_ARM_4[l] * c[l];
			    vzx += X_ARM_5[l] * c[l];
			    
			    pxy += Y_ARM_0[l] * c[l];
			    pyy += Y_ARM_1[l] * c[l];
			    pzy += Y_ARM_2[l] * c[l];
			    
			    vxy += Y_ARM_3[l] * c[l];
			    vyy += Y_ARM_4[l] * c[l];
			    vzy += Y_ARM_5[l] * c[l];
			    
			    pxz += Z_ARM_0[l] * c[l];
			    pyz += Z_ARM_1[l] * c[l];
			    pzz += Z_ARM_2[l] * c[l];
			    
			    vxz += Z_ARM_3[l] * c[l];
			    vyz += Z_ARM_4[l] * c[l];
			    vzz += Z_ARM_5[l] * c[l];

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

			float k_2_1 = vxx/s_4_4_4.data[0];
			float k_2_2 =  sigmax*px;
			wr_vec.data[2]= k_2_1 - k_2_2;            //vxx/rho[OPS_ACC4(0,0,0)] - sigmax*px;

			float k_5_1 = (pxx+pyx+pxz);
			float k_5_2 = k_5_1 *s_4_4_4.data[1];
			float k_5_3 = sigmax*vx;
			wr_vec.data[5]= k_5_2 - k_5_3;  //(pxx+pyx+pxz)*mu[OPS_ACC5(0,0,0)] - sigmax*vx;

			float k_3_1 = vyy/s_4_4_4.data[0];
			float k_3_2 = sigmay*py;
			wr_vec.data[3]= k_3_1 - k_3_2;  		  // vyy/rho[OPS_ACC4(0,0,0)] - sigmay*py;

			float k_6_1 = (pxy+pyy+pyz);
			float k_6_2 = k_6_1 *s_4_4_4.data[1];
			float k_6_3 = sigmay*vy;
			wr_vec.data[6]= k_6_2 - k_6_3;   //(pxy+pyy+pyz)*mu[OPS_ACC5(0,0,0)] - sigmay*vy;

			float k_4_1 = vzz/s_4_4_4.data[0];
			float k_4_2 = sigmaz*pz;
			wr_vec.data[4]= k_4_1  - k_4_2;  		  //vzz/rho[OPS_ACC4(0,0,0)] - sigmaz*pz;

			float k_7_1 = (pxz+pyz+pzz);
			float k_7_2 = k_7_1*s_4_4_4.data[1];
			float k_7_3 = sigmaz*vz;
			wr_vec.data[7]= k_7_2 - k_7_3;  //(pxz+pyz+pzz)*mu[OPS_ACC5(0,0,0)] - sigmaz*vz;

	  		wr_vec.data[0] = s_4_4_4.data[0];
	  		wr_vec.data[1] = s_4_4_4.data[1];

	  		struct dPath wr_vec_dt;
	  		// calc K dt
	  		wr_vec_dt.data[2] = wr_vec.data[2] * 0.1f;
	  		wr_vec_dt.data[5] = wr_vec.data[5] * 0.1f;
	  		
	  		wr_vec_dt.data[3] = wr_vec.data[3] * 0.1f;
	  		wr_vec_dt.data[6] = wr_vec.data[6] * 0.1f;
	  		
	  		wr_vec_dt.data[4] = wr_vec.data[4] * 0.1f;
	  		wr_vec_dt.data[7] = wr_vec.data[7] * 0.1f;

	  		wr_vec_dt.data[0] = s_4_4_4.data[0];
	  		wr_vec_dt.data[1] = s_4_4_4.data[1];

	  		// YY final 
	  		yy_final_vec.data[2] += wr_vec_dt.data[2] * 0.33333333333f;
	  		yy_final_vec.data[5] += wr_vec_dt.data[5] * 0.33333333333f;

	  		yy_final_vec.data[3] += wr_vec_dt.data[3] * 0.33333333333f;
	  		yy_final_vec.data[6] += wr_vec_dt.data[6] * 0.33333333333f;

	  		yy_final_vec.data[4] += wr_vec_dt.data[4] * 0.33333333333f;
	  		yy_final_vec.data[7] += wr_vec_dt.data[7] * 0.33333333333f;

	  		yy_final_vec.data[0] = wr_vec_dt.data[0];
	  		yy_final_vec.data[1] = wr_vec_dt.data[1];

	  		// calc Y temp
	  		struct dPath yyTmp_vec;
	  		yyTmp_vec.data[2] = yy_vec.data[2] + wr_vec_dt.data[2];
	  		yyTmp_vec.data[5] = yy_vec.data[5] + wr_vec_dt.data[5];
	  	
	  		yyTmp_vec.data[3] = yy_vec.data[3] + wr_vec_dt.data[3];
	  		yyTmp_vec.data[6] = yy_vec.data[6] + wr_vec_dt.data[6];
	  	
	  		yyTmp_vec.data[4] = yy_vec.data[4] + wr_vec_dt.data[4];
	  		yyTmp_vec.data[7] = yy_vec.data[7] + wr_vec_dt.data[7];

	  		yyTmp_vec.data[0] = s_4_4_4.data[0];
	  		yyTmp_vec.data[1] = s_4_4_4.data[1];




	  		bool change_cond1 = (idx < 0) || (idx >= sizex) || (idy < 0) ;
	  		bool change_cond2 = (idy >= sizey ) || (idz < 0) || (idz >= sizez);

	  		bool change_cond = change_cond1 || change_cond2;
			update_j = change_cond ? s_4_4_4: yyTmp_vec;
			yy_final_vec_write = change_cond ? s_4_4_4: yy_final_vec;




			bool cond_wr = (i >= ORDER) && ( i < grid_sizez + ORDER);
			if(cond_wr ) {
				pipeS::PipeAt<3>::write(update_j);
				pipeS::PipeAt<13>::write(yy_vec);
				pipeS::PipeAt<23>::write(yy_final_vec_write);
			}

			// move the cell block
			k_dum = k + 1;
		}
	});
    });
}
