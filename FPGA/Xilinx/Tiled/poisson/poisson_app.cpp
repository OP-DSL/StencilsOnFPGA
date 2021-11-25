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
* DISCLAIMED. IN NO EVENT SHALL Mike Giles BE LIABLE FOR ANY
* DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
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
#include "xcl2.hpp"
#include <chrono>
#include "omp.h"

int stencil_computation(float* current, float* next, int act_sizex, int act_sizey, int grid_size_x, int grid_size_y, int batches){
	for(int bat = 0; bat < batches; bat++){
		int offset = bat * grid_size_x* grid_size_y;
		for(int i = 0; i < act_sizey; i++){
		  for(int j = 0; j < act_sizex; j++){
			if(i == 0 || j == 0 || i == act_sizey -1  || j==act_sizex-1){
			  next[i*grid_size_x + j + offset] = current[i*grid_size_x + j + offset] ;
			} else {
			  next[i*grid_size_x + j + offset] = current[i*grid_size_x + j + offset]  * 0.5  + \
										   (current[(i-1)*grid_size_x + j+ offset] + current[(i+1)*grid_size_x + j+offset]) * 0.125f  + \
										   (current[i*grid_size_x + j+1+offset] + current[i*grid_size_x + j-1+offset]) * 0.125f ;
			}
		  }
		}
	}
    return 0;
}

double square_error(float* current, float* next, int act_sizex, int act_sizey, int grid_size_x, int grid_size_y, int batches){
  double sum = 0; 
  int count = 0;
  for(int bat = 0; bat < batches; bat++){
	  int offset = bat * grid_size_x* grid_size_y;
    for(int i = 0; i < act_sizey; i++){
      for(int j = 0; j < act_sizex; j++){
    	float val1 = next[i*grid_size_x + j+offset];
    	float val2 = current[i*grid_size_x + j+offset];
    	if(fabs(val1 -val2)/(fabs(val1)+fabs(val2)) > 0.00001 && (fabs(val1)+fabs(val2)) > 1.0e-5){
    		printf("i:%d j:%d %f %f \n",i,j, val1,val2);
    		count++;
    	}
        sum += next[i*grid_size_x + j+offset]*next[i*grid_size_x + j+offset] - current[i*grid_size_x + j+offset]*current[i*grid_size_x + j+offset];
      }
    }
  }
  printf("error count is %d\n", count);
    return sum;
}

int copy_grid(float* grid_s, float* grid_d, int act_sizex, int act_sizey, int grid_size_x, int grid_size_y, int batches){
    double sum = 0; 
    for(int bat = 0; bat < batches; bat++){
    	int offset = bat * grid_size_x* grid_size_y;
		for(int i = 0; i < act_sizey; i++){
		  for(int j = 0; j < act_sizex; j++){
			grid_d[i*grid_size_x + j+offset] = grid_s[i*grid_size_x + j+offset];
		  }
		}
    }
    return 0;
}


int initialise_grid(float* grid, int act_sizex, int act_sizey, int grid_size_x, int grid_size_y, int batches){
	for(int bat = 0; bat < batches; bat++){
	int offset = bat * grid_size_x* grid_size_y;
	  for(int i = 0; i < act_sizey; i++){
	  for(int j = 0; j < act_sizex; j++){
//		  grid[i*grid_size_x + j+offset] = i+j;
		  if(i == 0 || j == 0 || j == act_sizex -1  || i==act_sizey-1){
			float r = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
			grid[i*grid_size_x + j+offset] = r;
		  } else {
			grid[i*grid_size_x + j+offset] = 0;
		  }
		}
	  }
	}
  return 0;
}


// // OPS header file
// #define OPS_2D
// #include "ops_seq_v2.h"
// #include "user_types.h"
// #include "poisson_kernel.h"

/******************************************************************************
* Main program
*******************************************************************************/
int main(int argc, char **argv)
{
  /**-------------------------- Initialisation --------------------------**/

  // OPS initialisation
  // ops_init(argc,argv,1);


  //Mesh
  int logical_size_x = 20;
  int logical_size_y = 20;
  int ngrid_x = 1;
  int ngrid_y = 1;
  int n_iter = 10;
  int itertile = n_iter;
  int non_copy = 0;
  int batches = 1;
  unsigned short tile_size = 256;

  const char* pch;
  for ( int n = 1; n < argc; n++ ) {
    pch = strstr(argv[n], "-sizex=");
    if(pch != NULL) {
      logical_size_x = atoi ( argv[n] + 7 ); continue;
    }
    pch = strstr(argv[n], "-sizey=");
    if(pch != NULL) {
      logical_size_y = atoi ( argv[n] + 7 ); continue;
    }
    pch = strstr(argv[n], "-iters=");
    if(pch != NULL) {
      n_iter = atoi ( argv[n] + 7 ); continue;
    }
    pch = strstr(argv[n], "-itert=");
    if(pch != NULL) {
      itertile = atoi ( argv[n] + 7 ); continue;
    }
    pch = strstr(argv[n], "-batch=");
	if(pch != NULL) {
	  batches = atoi ( argv[n] + 7 ); continue;
	}
	pch = strstr(argv[n], "-tileX=");
	if(pch != NULL) {
		tile_size = atoi ( argv[n] + 7 ); continue;
	}
    pch = strstr(argv[n], "-non-copy");
    if(pch != NULL) {
      non_copy = 1; continue;
    }
  }

  logical_size_y = logical_size_y % 2 == 1 ? logical_size_y + 1: logical_size_y;
//  logical_size_y = (logical_size_y+2) % 4 != 0 ? ((logical_size_y+2)/4 + 1)*4 -2 : logical_size_y;

  printf("Grid: %dx%d in %dx%d blocks, %d iterations, %d tile height, %d batches\n",logical_size_x,logical_size_y,ngrid_x,ngrid_y,n_iter,itertile, batches);

  int act_sizex = logical_size_x + 2;
  int act_sizey = logical_size_y + 2;


  int grid_size_x = (act_sizex % 16) != 0 ? (act_sizex/16 +1) * 16 : act_sizex;
  int grid_size_y = act_sizey;

  int tile_count = 8;
  tile_size = tile_size % 16 == 0? tile_size : (tile_size/16 + 1) * 16;
  tile_size = tile_size > 64 ? tile_size : 64;

  unsigned short effective_tile_size= tile_size - 64;
  tile_count = grid_size_x % effective_tile_size != 0? grid_size_x / effective_tile_size + 1 : grid_size_x / effective_tile_size;
  
  int data_size_bytes = grid_size_x * grid_size_y * sizeof(float)*batches;
  float * grid_u1 = (float*)aligned_alloc(4096, data_size_bytes);
  float * grid_u2 = (float*)aligned_alloc(4096, data_size_bytes);

  float * grid_u1_d = (float*)aligned_alloc(4096, data_size_bytes);
  float * grid_u2_d = (float*)aligned_alloc(4096, data_size_bytes);

  unsigned int * tile = (unsigned int*)aligned_alloc(4096, 256*sizeof(unsigned int));
  int tile_c;
  int toltal_sizex = 0;
  for(int i = 0; i < tile_count; i++){
	  tile_c = i+1;
	  tile[i] = i* effective_tile_size  | (tile_size << 16);
	  if(i* effective_tile_size + tile_size >= grid_size_x){
		  tile[i] = i* effective_tile_size  | (grid_size_x - i* effective_tile_size) << 16;
		  toltal_sizex += (grid_size_x - i* effective_tile_size);
		  break;
	  }
	  toltal_sizex += tile_size;
  }
  printf("tile count is %d\n", tile_c);
//  tile[tile_count -1] = (tile_count -1) * effective_tile_size | ((grid_size_x - (tile_count -1) * effective_tile_size)  << 16);
//  if(tile[tile_count -1] <= 32){
//	  tile[tile_count -1]--;
//  }

  	tile_count = tile_c;

//  tile[0] = 0   | (128 << 16);
//  tile[1] = 64  | (128 << 16);
//  tile[2] = 128 | (128 << 16);
//  tile[3] = 192 | (128 << 16);
//  tile[4] = 256 | (128 << 16);
//  tile[5] = 320 | (128 << 16);
//  tile[6] = 384 | (128 << 16);
//  tile[7] = 448 | (64 << 16);
//  tile[5] = 320 | (128 << 16);

  initialise_grid(grid_u1, act_sizex, act_sizey, grid_size_x, grid_size_y, batches);
  copy_grid(grid_u1, grid_u1_d, act_sizex, act_sizey, grid_size_x, grid_size_y, batches);
  // stencil computation


   printf("\nI am here\n");
   fflush(stdout);

  //OPENCL HOST CODE AREA START

    auto binaryFile = argv[1];
    cl_int err;
    cl::Event event;

    auto devices = xcl::get_xil_devices();
    auto device = devices[0];

    OCL_CHECK(err, cl::Context context(device, NULL, NULL, NULL, &err));
    OCL_CHECK(
        err,
        cl::CommandQueue q(context, device, CL_QUEUE_PROFILING_ENABLE | CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, &err));
    OCL_CHECK(err,
              std::string device_name = device.getInfo<CL_DEVICE_NAME>(&err));



    //Create Program and Kernel
    auto fileBuf = xcl::read_binary_file(binaryFile);
    cl::Program::Binaries bins{{fileBuf.data(), fileBuf.size()}};
    devices.resize(1);
    auto start_p = std::chrono::high_resolution_clock::now();
    OCL_CHECK(err, cl::Program program(context, devices, bins, NULL, &err));
    OCL_CHECK(err, cl::Kernel krnl_Read_Write(program, "stencil_Read_Write", &err));
    OCL_CHECK(err, cl::Kernel krnl_slr0(program, "stencil_SLR0", &err));
    OCL_CHECK(err, cl::Kernel krnl_slr1(program, "stencil_SLR1", &err));
    OCL_CHECK(err, cl::Kernel krnl_slr2(program, "stencil_SLR2", &err));
    auto end_p = std::chrono::high_resolution_clock::now();
    auto dur_p = end_p -start_p;
    printf("time to program FPGA is %f\n", dur_p.count());


    //Allocate Buffer in Global Memory
    OCL_CHECK(err,
              cl::Buffer buffer_input(context,
                                      CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                                      data_size_bytes,
                                      grid_u1_d,
                                      &err));
    OCL_CHECK(err,
              cl::Buffer buffer_output(context,
                                       CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                                       data_size_bytes,
                                       grid_u2_d,
                                       &err));

    OCL_CHECK(err,
              cl::Buffer buffer_tile(context,
                                       CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
									   tile_count*sizeof(unsigned int),
                                       tile,
                                       &err));


    int read_write_offset = data_size_bytes/64;
    //Set the Kernel Arguments
    int narg = 0;
    OCL_CHECK(err, err = krnl_Read_Write.setArg(narg++, buffer_input));
    OCL_CHECK(err, err = krnl_Read_Write.setArg(narg++, buffer_input));
    OCL_CHECK(err, err = krnl_Read_Write.setArg(narg++, buffer_output));
    OCL_CHECK(err, err = krnl_Read_Write.setArg(narg++, buffer_output));
    OCL_CHECK(err, err = krnl_Read_Write.setArg(narg++, buffer_tile));
    OCL_CHECK(err, err = krnl_Read_Write.setArg(narg++, tile_count));
    OCL_CHECK(err, err = krnl_Read_Write.setArg(narg++, logical_size_x));
    OCL_CHECK(err, err = krnl_Read_Write.setArg(narg++, logical_size_y));
    OCL_CHECK(err, err = krnl_Read_Write.setArg(narg++, grid_size_x));
    OCL_CHECK(err, err = krnl_Read_Write.setArg(narg++, n_iter));

    narg = 0;
    OCL_CHECK(err, err = krnl_slr0.setArg(narg++, tile_count));
    OCL_CHECK(err, err = krnl_slr0.setArg(narg++, logical_size_x));
    OCL_CHECK(err, err = krnl_slr0.setArg(narg++, logical_size_y));
    OCL_CHECK(err, err = krnl_slr0.setArg(narg++, grid_size_x));
    OCL_CHECK(err, err = krnl_slr0.setArg(narg++, n_iter));

    narg = 0;
    OCL_CHECK(err, err = krnl_slr1.setArg(narg++, tile_count));
    OCL_CHECK(err, err = krnl_slr1.setArg(narg++, logical_size_x));
    OCL_CHECK(err, err = krnl_slr1.setArg(narg++, logical_size_y));
    OCL_CHECK(err, err = krnl_slr1.setArg(narg++, grid_size_x));
    OCL_CHECK(err, err = krnl_slr1.setArg(narg++, n_iter));

  	narg = 0;
  	OCL_CHECK(err, err = krnl_slr2.setArg(narg++, tile_count));
    OCL_CHECK(err, err = krnl_slr2.setArg(narg++, logical_size_x));
    OCL_CHECK(err, err = krnl_slr2.setArg(narg++, logical_size_y));
  	OCL_CHECK(err, err = krnl_slr2.setArg(narg++, grid_size_x));
  	OCL_CHECK(err, err = krnl_slr2.setArg(narg++, n_iter));

    //Copy input data to device global memory
    OCL_CHECK(err,
              err = q.enqueueMigrateMemObjects({buffer_input, buffer_tile},
                                               0 /* 0 means from host*/));

    uint64_t wtime = 0;
    uint64_t nstimestart, nstimeend;
    auto start = std::chrono::high_resolution_clock::now();




	//Launch the Kernel
    OCL_CHECK(err, err = q.enqueueTask(krnl_Read_Write));
	OCL_CHECK(err, err = q.enqueueTask(krnl_slr0));
	OCL_CHECK(err, err = q.enqueueTask(krnl_slr1));
	OCL_CHECK(err, err = q.enqueueTask(krnl_slr2));
	q.finish();


    auto finish = std::chrono::high_resolution_clock::now();
    OCL_CHECK(err,
              err = q.enqueueMigrateMemObjects({buffer_input},
                                               CL_MIGRATE_MEM_OBJECT_HOST));

    OCL_CHECK(err,
                  err = q.enqueueMigrateMemObjects({buffer_output},
                                                   CL_MIGRATE_MEM_OBJECT_HOST));

    q.finish();

  for(int itr = 0; itr < n_iter*60; itr++){
      stencil_computation(grid_u1, grid_u2, act_sizex, act_sizey, grid_size_x, grid_size_y, batches);
      stencil_computation(grid_u2, grid_u1, act_sizex, act_sizey, grid_size_x, grid_size_y, batches);
  }
    
    std::chrono::duration<double> elapsed = finish - start;

  printf("Runtime on FPGA is %f seconds\n", elapsed.count());
  double error = square_error(grid_u1, grid_u1_d, act_sizex, act_sizey, grid_size_x, grid_size_y, batches);
  float bandwidth = (act_sizex * act_sizey * sizeof(float) * 4.0 * n_iter * 27)/(elapsed.count() * 1000 * 1000 * 1000);
  float logic_bandwidth = (toltal_sizex * act_sizey * sizeof(float) * 4.0 * n_iter)/(elapsed.count() * 1000 * 1000 * 1000);
  printf("\nMean Square error is  %f\n\n", error/(logical_size_x * logical_size_y));
  printf("\nBandwidth is %f \t %f\n", bandwidth,logic_bandwidth);








//  printf("\nBandwidth prof is %f\n", bandwidth_prof);

//  for(int i = 0; i < act_sizey; i++){
//    for(int j = 0; j < act_sizex; j++){
//        printf("%f ", grid_u1_d[i*grid_size_x + j]);
//    }
//    printf("\n");
//  }
//
//  printf("\ngolden\n\n");
//  for(int i = 0; i < act_sizey; i++){
//    for(int j = 0; j < act_sizex; j++){
//        printf("%f ", grid_u1[i*grid_size_x + j]);
//    }
//    printf("\n");
//  }


  free(grid_u1);
  free(grid_u2);
  free(grid_u1_d);
  free(grid_u2_d);
  free(tile);

  return 0;
}
