/*
* Open source copyright declaration based on BSD open source template:
* http://www.opensource.org/licenses/bsd-license.php
*
* This file is part of the OPS distribution.
*
* Copyright (c) 2013, Mike Giles and others. Please see the AUTHORS file in
* the main source directory for a full list of copyright holders.
* All rights reserved.
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions are met:
* * Redistributions of source code must retain the above copyright
* notice, this list of conditions and the following disclaimer.
* * Redistributions in binary form must reproduce the above copyright
* notice, this list of conditions and the following disclaimer in the
* documentation and/or other materials provided with the distribution.
* * The name of Mike Giles may not be used to endorse or promote products
* derived from this software without specific prior written permission.
*
* THIS SOFTWARE IS PROVIDED BY Mike Giles ''AS IS'' AND ANY
* EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
* WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
* DISCLAIMED. IN NO EVENT SHALL Mike Giles BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
* (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
* LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
* ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
* (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
* SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

/** @Test application for multi-block functionality
  * @author Gihan Mudalige, Istvan Reguly
  */

// standard headers
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

float dx,dy,dz;
int pml_width;
int half, order, nx, ny, nz;
#include "./coeffs/coeffs8.h"
// OPS header file
#define OPS_3D
#include "ops_seq_v2.h"

#include "poisson_kernel.h"

void derivs(int ngrid_x, int ngrid_y, int ngrid_z, int* sizes, int disps[], ops_block blocks, ops_stencil S3D_000, ops_stencil S3D_big_sten,  ops_dat rho, ops_dat mu, ops_dat yy, ops_dat dyy);

void calc_ytemp(int ngrid_x, int ngrid_y, int ngrid_z, int* sizes, int disps[], ops_block blocks, ops_stencil S3D_000, float* dt, ops_dat yy, ops_dat kk, ops_dat ytemp);

void calc_ytemp2(int ngrid_x, int ngrid_y, int ngrid_z, int* sizes, int disps[], ops_block blocks, ops_stencil S3D_000, float* dt, ops_dat yy, ops_dat kk, ops_dat ytemp);

void final_update(int ngrid_x, int ngrid_y, int ngrid_z, int* sizes, int disps[], ops_block blocks, ops_stencil S3D_000, float* dt, ops_dat k1, ops_dat k2, ops_dat k3, ops_dat k4, ops_dat yy);

	      
/******************************************************************************
* Main program
*******************************************************************************/
int main(int argc,  char **argv)
{
  /**-------------------------- Initialisation --------------------------**/

  // OPS initialisation  
  ops_init(argc,argv,1);


  //Mesh
  int logical_size_x = 16;
  int logical_size_y = 16;
  int logical_size_z = 16;
  nx = logical_size_x;
  ny = logical_size_y;
  nz = logical_size_z;
  int ngrid_x = 1;
  int ngrid_y = 1;
  int ngrid_z = 1;
  int n_iter = 100;
  int itertile = n_iter;
  int non_copy = 0;

  int num_systems = 100;

  const char* pch;
  printf(" argc = %d\n",argc);
  for ( int n = 1; n < argc; n++ ) {
    pch = strstr(argv[n], "-sizex=");
    if(pch != NULL) {
      logical_size_x = atoi ( argv[n] + 7 ); continue;
    }
    pch = strstr(argv[n], "-sizey=");
    if(pch != NULL) {
      logical_size_y = atoi ( argv[n] + 7 ); continue;
    }
    pch = strstr(argv[n], "-sizez=");
    if(pch != NULL) {
      logical_size_z = atoi ( argv[n] + 7 ); continue;
    }
    pch = strstr(argv[n], "-iters=");
    if(pch != NULL) {
      n_iter = atoi ( argv[n] + 7 ); continue;
    }
    pch = strstr(argv[n], "-itert=");
    if(pch != NULL) {
      itertile = atoi ( argv[n] + 7 ); continue;
    }
    pch = strstr(argv[n], "-non-copy");
    if(pch != NULL) {
      non_copy = 1; continue;
    }
    pch = strstr(argv[n], "-batch=");
    if(pch != NULL) {
      num_systems = atoi ( argv[n] + 7 ); continue;
    }
  }

  // logical_size_x = 16;
  // logical_size_y = 16;
  // logical_size_z = 16;
  nx = logical_size_x;
  ny = logical_size_y;
  nz = logical_size_z;
  ngrid_x = 1;
  ngrid_y = 1;
  ngrid_z = 1;
  // n_iter = 1;
  itertile = n_iter;
  non_copy = 0;

  ops_printf("Grid: %dx%dx%d , %d iterations, %d tile height\n",logical_size_x,logical_size_y,logical_size_z,n_iter,itertile);
  dx = 0.005;
  dy = 0.005;
  dz = 0.005;
  pml_width = 10;
  ops_decl_const("dx",1,"float",&dx);
  ops_decl_const("dy",1,"float",&dy);
  ops_decl_const("dz",1,"float",&dz);
  ops_decl_const("nx",1,"int",&nx);
  ops_decl_const("ny",1,"int",&ny);
  ops_decl_const("nz",1,"int",&nz);
  ops_decl_const("pml_width",1,"int",&pml_width);
  //int ncoeffs = (ORDER+1)*(ORDER+1);
  //ops_decl_const("coeffs",ncoeffs,"double",&coeffs[0][0]);
  half = HALF;
  ops_decl_const("half",1,"int",&half);
  order = ORDER;
  ops_decl_const("order",1,"int",&order);

  //declare block
  char buf[50];
  sprintf(buf,"block");
  ops_block blocks = ops_decl_block_batch(3,"block", num_systems, OPS_BATCHED);
  printf(" HERE \n");
  
  //declare stencils
  int s3D_000[]         = {0,0,0};
  ops_stencil S3D_000 = ops_decl_stencil( 3, 1, s3D_000, "000");
  int s3D_big_sten[3*3*(2*ORDER+1)];
  int is = 0;
  for (int ix=-HALF;ix<=HALF;ix++) {
    printf("ix = %d\n",ix);
    s3D_big_sten[is] = ix;
    is = is + 1;
    s3D_big_sten[is] = 0;
    is = is + 1;
    s3D_big_sten[is] = 0;
    is = is + 1;
  }
  for (int ix=-HALF;ix<=HALF;ix++) {
    s3D_big_sten[is] = 0;
    is = is + 1;
    s3D_big_sten[is] = ix;
    is = is + 1;
    s3D_big_sten[is] = 0;
    is = is + 1;
  }
  for (int ix=-HALF;ix<=HALF;ix++) {
    s3D_big_sten[is] = 0;
    is = is + 1;
    s3D_big_sten[is] = 0;
    is = is + 1;
    s3D_big_sten[is] = ix;
    is = is + 1;
  }
  ops_stencil S3D_big_sten = ops_decl_stencil( 3, 3*(2*ORDER+1), s3D_big_sten, "big_sten");

  printf(" HERE2 \n");
  

  //declare datasets
  int d_p[3] = {HALF,HALF,HALF}; //max halo depths for the dat in the possitive direction
  int d_m[3] = {-HALF,-HALF,-HALF}; //max halo depths for the dat in the negative direction
  int base[3] = {0,0,0};
  int uniform_size[3] = {(logical_size_x-1)/ngrid_x+1,(logical_size_y-1)/ngrid_y+1,(logical_size_z-1)/ngrid_z+1};
  float* temp = NULL;
  int *sizes = (int*)malloc(3*sizeof(int));
  int *disps = (int*)malloc(3*sizeof(int));

  printf(" HERE 3\n");
  
  int size[3] = {uniform_size[0], uniform_size[1], uniform_size[2]};
  if (size[0]>logical_size_x) size[0] = logical_size_x;
  if (size[1]>logical_size_y) size[1] = logical_size_y;
  if (size[2]>logical_size_z) size[2] = logical_size_z;
	
  sprintf(buf,"coordx");
  ops_dat coordx = ops_decl_dat(blocks, 1, size, base, d_m, d_p, temp, "float", buf);
  sprintf(buf,"coordy");
  ops_dat coordy = ops_decl_dat(blocks, 1, size, base, d_m, d_p, temp, "float", buf);
  sprintf(buf,"coordz");
  ops_dat coordz= ops_decl_dat(blocks, 1, size, base, d_m, d_p, temp, "float", buf);
  sprintf(buf,"rho");
  ops_dat rho = ops_decl_dat(blocks, 1, size, base, d_m, d_p, temp, "float", buf);
  sprintf(buf,"mu");
  ops_dat mu = ops_decl_dat(blocks, 1, size, base, d_m, d_p, temp, "float", buf);
  sprintf(buf,"yy");
  ops_dat yy = ops_decl_dat(blocks, 6, size, base, d_m, d_p, temp, "float", buf);
  sprintf(buf,"yy_new");
  ops_dat yy_new = ops_decl_dat(blocks, 6, size, base, d_m, d_p, temp, "float", buf);
  sprintf(buf,"ytemp");
  ops_dat ytemp = ops_decl_dat(blocks, 6, size, base, d_m, d_p, temp, "float", buf);
  sprintf(buf,"k1");
  ops_dat k1 = ops_decl_dat(blocks, 6, size, base, d_m, d_p, temp, "float", buf);
  sprintf(buf,"k2");
  ops_dat k2 = ops_decl_dat(blocks, 6, size, base, d_m, d_p, temp, "float", buf);
  sprintf(buf,"k3");
  ops_dat k3 = ops_decl_dat(blocks, 6, size, base, d_m, d_p, temp, "float", buf);
  sprintf(buf,"k4");
  ops_dat k4 = ops_decl_dat(blocks, 6, size, base, d_m, d_p, temp, "float", buf);
  
  sizes[0] = size[0];
  sizes[1] = size[1];
  sizes[2] = size[2];
  disps[0] = 0;
  disps[1] = 0;
  disps[2] = 0;

  printf(" HERE 4\n");

  /*
  /* THERE ARE NO HALOS IF ngrid_x = 1, ngrix_y = 1, and ngrid_z = 1, SO I'M IGNORING 
     THIS HERE AND LEAVING THE CODE AS IS*/
  /*
  ops_halo *halos = (ops_halo *)malloc(2*(ngrid_x*(ngrid_y-1)+(ngrid_x-1)*ngrid_y*sizeof(ops_halo)));
  int off = 0;
  for (int j = 0; j < ngrid_y; j++) {
    for (int i = 0; i < ngrid_x; i++) {
      if (i > 0) {
        int halo_iter[] = {1,sizes[2*(i+ngrid_x*j)+1]};
        int base_from[] = {sizes[2*(i-1+ngrid_x*j)]-1,0};
        int base_to[] = {-1,0};
        int dir[] = {1,2};

        halos[off++] = ops_decl_halo(mu[i-1+ngrid_x*j], mu[i+ngrid_x*j], halo_iter, base_from, base_to, dir, dir);
        base_from[0] = 0; base_to[0] = sizes[2*(i+ngrid_x*j)];
        halos[off++] = ops_decl_halo(mu[i+ngrid_x*j], mu[i-1+ngrid_x*j], halo_iter, base_from, base_to, dir, dir);
      }
      if (j > 0) {
        int halo_iter[] = {sizes[2*(i+ngrid_x*j)],1};
        int base_from[] = {0,sizes[2*(i+ngrid_x*(j-1))+1]-1};
        int base_to[] = {0,-1};
        int dir[] = {1,2};

        halos[off++] = ops_decl_halo(mu[i+ngrid_x*(j-1)], mu[i+ngrid_x*j], halo_iter, base_from, base_to, dir, dir);
        base_from[1] = 0; base_to[1] = sizes[2*(i+ngrid_x*j)+1];
        halos[off++] = ops_decl_halo(mu[i+ngrid_x*j], mu[i+ngrid_x*(j-1)], halo_iter, base_from, base_to, dir, dir);
      }
    }
  }
  if (off != 2*(ngrid_x*(ngrid_y-1)+(ngrid_x-1)*ngrid_y)) printf("Something is not right\n");
  ops_halo_group u_halos = ops_decl_halo_group(off,halos);
  */
  
  ops_partition("");
  ops_checkpointing_init("check.h5", 5.0, 0);
  ops_diagnostic_output();
  /**-------------------------- Computations --------------------------**/


  double ct0, ct1, et0, et1;
  ops_timers(&ct0, &et0);

  ops_par_loop_blocks_all(num_systems);
	
  printf(" HERE 5\n");
  
  //populate density, bulk modulus, velx, vely, velz, and boundary conditions
	
  int iter_range[] = {0,sizes[0],0,sizes[1],0,sizes[2]};
	
  ops_par_loop(rtm_kernel_populate, "rtm_kernel_populate", blocks, 3, iter_range,
	       ops_arg_gbl(&disps[0], 1, "int", OPS_READ),
	       ops_arg_gbl(&disps[1], 1, "int", OPS_READ),
	       ops_arg_gbl(&disps[2], 1, "int", OPS_READ),
	       ops_arg_idx(),
	       ops_arg_dat(rho, 1, S3D_000, "float", OPS_WRITE),
	       ops_arg_dat(mu, 1, S3D_000, "float", OPS_WRITE),
	       ops_arg_dat(yy, 6, S3D_000, "float", OPS_WRITE));
		
  printf(" DONE populate\n");
  

  double it0, it1;
  ops_timers(&ct0, &it0);
//#ifdef NOT_NOW  
  for (int iter = 0; iter < n_iter; iter++) {
    //if (ngrid_x>1 || ngrid_y>1 || ngrid_z>1) ops_halo_transfer(u_halos);
    // if (iter%itertile == 0) ops_execute();

    /* The following is 4th order Runga-Kutta */
    float dt = 0.1; //=sqrt(mu/rho);
    // calc dydt (store it in k1)
    // printf(" CALLING FIRST derivs \n");
    derivs(ngrid_x, ngrid_y, ngrid_z, sizes, disps, blocks, S3D_000, S3D_big_sten,
	   rho, mu, yy, k1);

    // printf(" DONE derivs\n");
    

    //k1 = dt*k1
    //ytemp = yy + k1/2.
    calc_ytemp(ngrid_x, ngrid_y, ngrid_z, sizes, disps, blocks, S3D_000, &dt,
	       yy, k1, ytemp);



    // ops_print_dat_to_txtfile(k1, "k1.txt");
    
    // printf(" DONE calc_ytemp\n");

    // calc dytemp/dt store it in k2
    derivs(ngrid_x, ngrid_y, ngrid_z, sizes, disps, blocks, S3D_000, S3D_big_sten,
	   rho, mu, ytemp, k2);
    //k2 = dt*k2 parallel_loop
    //ytemp = yy + k2/2.
    calc_ytemp(ngrid_x, ngrid_y, ngrid_z, sizes, disps, blocks, S3D_000, &dt,
	       yy, k2, ytemp);

    //calc dytemp/dt (stored it in k3)
    derivs(ngrid_x, ngrid_y, ngrid_z, sizes, disps, blocks, S3D_000, S3D_big_sten,
	   rho, mu, ytemp, k3);
    //k3 = k3*dt
    //ytemp = yy + k3
    calc_ytemp2(ngrid_x, ngrid_y, ngrid_z, sizes, disps, blocks, S3D_000, &dt,
	        yy, k3, ytemp);

    

    //calc dytemp/dt, stored in k4
    derivs(ngrid_x, ngrid_y, ngrid_z, sizes, disps, blocks, S3D_000, S3D_big_sten,
	   rho, mu, ytemp, k4);


    //k4 = dt*k4
    //yy = yy + k1/6. + k2/3. + k3/3. + k4/6.
    final_update(ngrid_x, ngrid_y, ngrid_z, sizes, disps, blocks, S3D_000, &dt,
	         k1, k2, k3, k4, yy);

   



	     
  }
//#endif
  // ops_execute();
  ops_timers(&ct0, &it1);
   // ops_print_dat_to_txtfile(yy, "yy.txt");

  //ops_print_dat_to_txtfile(u[0], "poisson.dat");
  //ops_print_dat_to_txtfile(ref[0], "poisson.dat");
  //exit(0);


  // free(coordx);
  // free(coordy);
  // free(coordz);
  // free(rho);
  // free(mu);
  // free(yy);
  // free(ytemp);
  // free(k1);
  // free(k2);
  // free(k3);
  // free(k4);


  ops_timers(&ct1, &et1);
  ops_timing_output(stdout);
  ops_printf("\nTotal Wall time %lf\n",et1-et0);

  ops_printf("%lf\n",it1-it0);
  
  ops_printf(" DONE !!!\n");

  ops_exit();
}

void derivs(int ngrid_x, int ngrid_y, int ngrid_z, int* sizes, int disps[], ops_block blocks, ops_stencil S3D_000, ops_stencil S3D_big_sten, ops_dat rho, ops_dat mu, ops_dat yy, ops_dat dyy) {
  
  int iter_range[] = {0,sizes[0],0,sizes[1],0,sizes[2]};
  ops_par_loop(fd3d_pml_kernel, "fd3d_pml_kernel", blocks, 3, iter_range,
	       ops_arg_gbl(&disps[0], 1, "int", OPS_READ),
	       ops_arg_gbl(&disps[1], 1, "int", OPS_READ),
	       ops_arg_gbl(&disps[2], 1, "int", OPS_READ),
	       ops_arg_idx(),
	       ops_arg_dat(rho, 1, S3D_000, "float", OPS_READ),
	       ops_arg_dat(mu, 1, S3D_000, "float", OPS_READ),
	       ops_arg_dat(yy, 6, S3D_big_sten, "float", OPS_READ),
	       ops_arg_dat(dyy, 6, S3D_000, "float", OPS_WRITE)
	       );
}

void calc_ytemp(int ngrid_x, int ngrid_y, int ngrid_z, int* sizes, int disps[], ops_block blocks, ops_stencil S3D_000, float* dt, ops_dat yy, ops_dat kk, ops_dat ytemp) {

  int iter_range[] = {0,sizes[0],0,sizes[1],0,sizes[2]};
  ops_par_loop(calc_ytemp_kernel, "calc_ytemp_kernel", blocks, 3, iter_range,
               ops_arg_gbl(&disps[0], 1, "int", OPS_READ),
	       ops_arg_gbl(&disps[1], 1, "int", OPS_READ),
	       ops_arg_gbl(&disps[2], 1, "int", OPS_READ),
	       ops_arg_idx(),
	       ops_arg_gbl(dt, 1, "float", OPS_READ),	       
	       ops_arg_dat(yy, 6, S3D_000, "float", OPS_READ),
	       ops_arg_dat(kk, 6, S3D_000, "float", OPS_RW),
	       ops_arg_dat(ytemp, 6, S3D_000, "float", OPS_WRITE)
	       );
}

void calc_ytemp2(int ngrid_x, int ngrid_y, int ngrid_z, int* sizes, int disps[], ops_block blocks, ops_stencil S3D_000, float* dt, ops_dat yy, ops_dat kk, ops_dat ytemp) {

  int iter_range[] = {0,sizes[0],0,sizes[1],0,sizes[2]};
  ops_par_loop(calc_ytemp2_kernel, "calc_ytemp2_kernel", blocks, 3, iter_range,
               ops_arg_gbl(&disps[0], 1, "int", OPS_READ),
	       ops_arg_gbl(&disps[1], 1, "int", OPS_READ),
	       ops_arg_gbl(&disps[2], 1, "int", OPS_READ),
	       ops_arg_idx(),
	       ops_arg_gbl(dt, 1, "float", OPS_READ),	       
	       ops_arg_dat(yy, 6, S3D_000, "float", OPS_READ),
	       ops_arg_dat(kk, 6, S3D_000, "float", OPS_RW),
	       ops_arg_dat(ytemp, 6, S3D_000, "float", OPS_WRITE)
	       );
}

void final_update(int ngrid_x, int ngrid_y, int ngrid_z, int* sizes, int disps[], ops_block blocks, ops_stencil S3D_000, float* dt, ops_dat k1, ops_dat k2, ops_dat k3, ops_dat k4, ops_dat yy) {

  int iter_range[] = {0,sizes[0],0,sizes[1],0,sizes[2]};
  ops_par_loop(final_update_kernel, "final_update_kernel", blocks, 3, iter_range,
         ops_arg_gbl(&disps[0], 1, "int", OPS_READ),
	       ops_arg_gbl(&disps[1], 1, "int", OPS_READ),
	       ops_arg_gbl(&disps[2], 1, "int", OPS_READ),
	       ops_arg_idx(),
	       ops_arg_gbl(dt, 1, "float", OPS_READ),	       
	       ops_arg_dat(k1, 6, S3D_000, "float", OPS_READ),
	       ops_arg_dat(k2, 6, S3D_000, "float", OPS_READ),
	       ops_arg_dat(k3, 6, S3D_000, "float", OPS_READ),
	       ops_arg_dat(k4, 6, S3D_000, "float", OPS_RW),
	       ops_arg_dat(yy, 6, S3D_000, "float", OPS_RW)
	       );
}  
