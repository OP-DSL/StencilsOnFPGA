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
#include <ratio>
#include <ctime>




int stencil_computation(float* current, float* next, int act_sizex, int act_sizey, int act_sizez, int grid_size_x, int grid_size_y, int grid_size_z){
    for(int i = 0; i < act_sizez; i++){
      for(int j = 0; j < act_sizey; j++){
        for(int k = 0; k < act_sizex; k++){
          if(i == 0 || j == 0 || k ==0 || i == act_sizez -1  || j==act_sizey-1 || k == act_sizex -1){
            next[i*grid_size_x*grid_size_y + j*grid_size_x + k] = current[i*grid_size_x*grid_size_y + j*grid_size_x + k] ;
          } else {
            next[i*grid_size_x*grid_size_y + j*grid_size_x + k] =   current[(i-1)*grid_size_x*grid_size_y + (j)*grid_size_x + (k)] * (0.01)  + \
																	current[(i+1)*grid_size_x*grid_size_y + (j)*grid_size_x + (k)] * (0.02)  + \
																	current[(i)*grid_size_x*grid_size_y + (j-1)*grid_size_x + (k)] * (0.03)  + \
																	current[(i)*grid_size_x*grid_size_y + (j+1)*grid_size_x + (k)] * (0.04)  + \
																	current[(i)*grid_size_x*grid_size_y + (j)*grid_size_x + (k-1)] * (0.05)  + \
																	current[(i)*grid_size_x*grid_size_y + (j)*grid_size_x + (k+1)] * (0.06)  + \
																	current[(i)*grid_size_x*grid_size_y + (j)*grid_size_x + (k)] * (0.79) ;
          }
        }
      }
    }
    return 0;
}

double square_error(float* current, float* next, int act_sizex, int act_sizey, int act_sizez, int grid_size_x, int grid_size_y, int grid_size_z){
    double sum = 0;
    int count = 0;
    for(int i = 0; i < act_sizez; i++){
      for(int j = 0; j < act_sizey; j++){
    	int flag = 0;
        for(int k = 0; k < act_sizex; k++){
          float val1 = next[i*grid_size_x*grid_size_y + j*grid_size_x+k];
          float val2 = current[i*grid_size_x*grid_size_y + j*grid_size_x+k];
          sum +=  val1*val1 - val2*val2;
          if((fabs(val1 -val2)/(fabs(val1) + fabs(val2))) > 0.000001 && (fabs(val1) + fabs(val2)) > 0.000001 && i == 1){
        	  printf("z:%d y:%d x:%d Val1:%f val2:%f\n", i, j, k, val1, val2);
        	  count++;
        	  flag = 1;
          }
        }
//        if(flag == 1){
//        	 printf("z:%d y:%d \n", i, j);
//        }
      }
    }
    printf("Error count is %d\n", count);
    return sum;
}

int copy_grid(float* grid_s, float* grid_d, unsigned int grid_size){
    memcpy(grid_d, grid_s, grid_size);
    return 0;
}


int initialise_grid(float* grid, unsigned int act_sizex, unsigned int act_sizey, unsigned int act_sizez, unsigned int grid_size_x, unsigned int grid_size_y, unsigned int grid_size_z){
  for(unsigned int i = 0; i < act_sizez; i++){
    for(unsigned int j = 0; j < act_sizey; j++){
      for(unsigned int k = 0; k < act_sizex; k++){
    	  unsigned int index = i*grid_size_x*grid_size_y + j * grid_size_x + k;
//        if(i == 0 || j == 0 || k == 0 || i == act_sizez -1  || j==act_sizey-1 || k == act_sizex-1 ){
          float r = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
          grid[index] = r;
//        } else {
//          grid[index] = 0;
//        }
      }
    }
  }
  return 0;
}

int split_grid(float* grid, float** grid8, unsigned int act_sizex, unsigned int act_sizey, unsigned int act_sizez, unsigned int grid_size_x, unsigned int grid_size_y, unsigned int grid_size_z){
  for(unsigned int i = 0; i < act_sizez; i++){
    for(unsigned int j = 0; j < act_sizey; j++){
      for(unsigned int k = 0; k < grid_size_x/128; k++){
    	  unsigned int index = i*grid_size_x*grid_size_y + j * grid_size_x + k*128;
    	  unsigned int index8 = i*grid_size_x/8*grid_size_y + j * grid_size_x/8 + k*16;
    	  memcpy(&grid8[0][index8], &grid[index], 64);
    	  memcpy(&grid8[1][index8], &grid[index+16], 64);
    	  memcpy(&grid8[2][index8], &grid[index+32], 64);
    	  memcpy(&grid8[3][index8], &grid[index+48], 64);

    	  memcpy(&grid8[4][index8], &grid[index+64], 64);
    	  memcpy(&grid8[5][index8], &grid[index+80], 64);
    	  memcpy(&grid8[6][index8], &grid[index+96], 64);
    	  memcpy(&grid8[7][index8], &grid[index+112], 64);

      }
    }
  }
  return 0;
}



int merge_grid(float** grid8, float* grid, unsigned int act_sizex, unsigned int act_sizey, unsigned int act_sizez, unsigned int grid_size_x, unsigned int grid_size_y, unsigned int grid_size_z){
  for(unsigned int i = 0; i < act_sizez; i++){
    for(unsigned int j = 0; j < act_sizey; j++){
      for(unsigned int k = 0; k < grid_size_x/128; k++){
    	  unsigned int index = i*grid_size_x*grid_size_y + j * grid_size_x + k*128;
    	  unsigned int index8 = i*grid_size_x/8*grid_size_y + j * grid_size_x/8 + k*16;
    	  memcpy(&grid[index], &grid8[0][index8], 64);
    	  memcpy(&grid[index+16], &grid8[1][index8], 64);
    	  memcpy(&grid[index+32], &grid8[2][index8], 64);
    	  memcpy(&grid[index+48], &grid8[3][index8], 64);

    	  memcpy(&grid[index+64], &grid8[4][index8], 64);
    	  memcpy(&grid[index+80], &grid8[5][index8], 64);
    	  memcpy(&grid[index+96], &grid8[6][index8], 64);
    	  memcpy(&grid[index+112], &grid8[7][index8], 64);
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
  int logical_size_z = 20;
  int ngrid_x = 1;
  int ngrid_y = 1;
  int n_iter = 10;
  int itertile = n_iter;
  int non_copy = 0;
  int batches = 1;
  unsigned short tilex_size = 32;
  unsigned short tiley_size = 32;
  unsigned short offset_x = 0;

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
    pch = strstr(argv[n], "-batch=");
	if(pch != NULL) {
	  batches = atoi ( argv[n] + 7 ); continue;
	}
	pch = strstr(argv[n], "-tileX=");
	if(pch != NULL) {
		tilex_size = atoi ( argv[n] + 7 ); continue;
	}
	pch = strstr(argv[n], "-tileY=");
	if(pch != NULL) {
		tiley_size = atoi ( argv[n] + 7 ); continue;
	}

	pch = strstr(argv[n], "-offsetX=");
	if(pch != NULL) {
		offset_x = atoi ( argv[n] + 9 ); continue;
	}
    pch = strstr(argv[n], "-non-copy");
    if(pch != NULL) {
      non_copy = 1; continue;
    }
  }

  logical_size_y = logical_size_y % 2 == 1 ? logical_size_y + 1: logical_size_y;
//  logical_size_y = (logical_size_y+2) % 4 != 0 ? ((logical_size_y+2)/4 + 1)*4 -2 : logical_size_y;

//  logical_size_x = 8192;
  printf("Grid: %dx%dx%d in %dx%d blocks, %d iterations, %d tile height, %d batches\n",logical_size_x,logical_size_y,logical_size_z, ngrid_x,ngrid_y,n_iter,itertile, batches);

  int act_sizex = logical_size_x + 2;
  int act_sizey = logical_size_y + 2;
  int act_sizez = logical_size_z + 2;


  tilex_size = tilex_size % 128 == 0? tilex_size : (tilex_size/128 + 1) * 128;

  int act_sizex_32 = (act_sizex% 32 != 0 ? (act_sizex/32 + 1)* 32 : act_sizex);
  int grid_size_x = (act_sizex % 128) != 0 ? (act_sizex/128 +1) * 128 : act_sizex;
  grid_size_x = grid_size_x + 128;
  int grid_size_y = act_sizey;
  int grid_size_z = act_sizez;




//  unsigned short effective_tile_size= tile_size - 0;
  unsigned int data_size_bytes = grid_size_x * (grid_size_y) * grid_size_z *sizeof(float)*batches;
  unsigned int data_size_bytes_div8 = data_size_bytes/8;
  float * grid_u1 = (float*)aligned_alloc(4096, data_size_bytes);
  float * grid_u2 = (float*)aligned_alloc(4096, data_size_bytes);

  float * grid_u1_d = (float*)aligned_alloc(4096, data_size_bytes);
  float * grid_u2_d = (float*)aligned_alloc(4096, data_size_bytes);

  float* grid_u1_d8[8];
  float* grid_u2_d8[8];
  for(int i = 0; i < 8; i++){
	  grid_u1_d8[i] = (float*)aligned_alloc(4096, data_size_bytes_div8);
	  grid_u2_d8[i] = (float*)aligned_alloc(4096, data_size_bytes_div8);
  }



  if(grid_u1 == NULL || grid_u2 == NULL || grid_u1_d == NULL || grid_u2_d == NULL){
	  printf("Error in allocating memory\n");
	  return -1;
  }

  // Tiling
  const int tile_max_count = 256;
  unsigned int * tile = (unsigned int*)aligned_alloc(4096, (tile_max_count)*sizeof(unsigned int));






  // tiling on x dimension
  int tilex_c;
  int toltal_sizex = 0;
  int effective_tilex_size = tilex_size - 32;
  for(int i = 0; i < tile_max_count; i++){
	  tilex_c = i+1;
	  tile[i] = i* effective_tilex_size  | (tilex_size << 16);
	  if(i* effective_tilex_size + tilex_size >= act_sizex_32){
		  int lastx_size = act_sizex_32 - i* effective_tilex_size;
		  lastx_size = lastx_size % 128 == 0 ? lastx_size : (lastx_size/128 +1)*128;
		  int offset_x = act_sizex_32-lastx_size > 0 ? act_sizex_32-lastx_size : 0;
		  tile[i] = offset_x | lastx_size << 16;
		  toltal_sizex += lastx_size;
		  break;
	  }
	  toltal_sizex += tilex_size;
  }





  int tilex_count = tilex_c;
  printf("Grid_size_y is %d\n", grid_size_y);

  // tiling on y dimension
  tiley_size = (tiley_size < 8 ? 8 : tiley_size);
  int tiley_c;
  int toltal_sizey = 0;
  int effective_tiley_size = tiley_size-6;




  for(int i = 0; i < tile_max_count; i++){
	  tiley_c = i+1;
	  tile[i+tilex_count] = i* effective_tiley_size  | (tiley_size << 16);
	  if(i* effective_tiley_size + tiley_size >= grid_size_y){
		  int lasty_size = grid_size_y - i* effective_tiley_size;
		  tile[i+tilex_count] = (grid_size_y - lasty_size)  | lasty_size << 16;
		  toltal_sizey += lasty_size;
		  break;
	  }
	  toltal_sizey += tiley_size;
  }

  int tiley_count = tiley_c;

//
  int tile_count = tilex_count + tiley_count;
  int total_plane_sizeR = 0, total_plane_sizeW = 0, total_plane_size;
  for(int i = 0; i < tiley_count; i++){
	  for(int j = 0; j < tilex_count; j++){
//		  printf("Tilex:%d TileY:%d\n", (tile[j] >> 16), (tile[i+tilex_count] >> 16));
		  total_plane_sizeR += (tile[j] >> 16) * (tile[i+tilex_count] >> 16);
		  total_plane_sizeW += ((tile[j] >> 16)) * ((tile[i+tilex_count] >> 16));
	  }
  }
  total_plane_size = total_plane_sizeR + total_plane_sizeW;
  printf("tilex_count:%d tiley_count:%d tile_count:%d\n", tilex_count, tiley_count, tiley_count*tilex_count);




  printf("\n before initialise grid\n");
  fflush(stdout);

  initialise_grid(grid_u1, act_sizex, act_sizey, act_sizez, grid_size_x, grid_size_y, grid_size_z);
  copy_grid(grid_u1, grid_u1_d, data_size_bytes);
  split_grid(grid_u1, grid_u1_d8, act_sizex, act_sizey, act_sizez, grid_size_x, grid_size_y, grid_size_z);


  //OPENCL HOST CODE AREA START

    auto binaryFile = argv[1];
    cl_int err;


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
    OCL_CHECK(err, cl::Kernel krnl_stencil_Read_Write(program, "stencil_Read_Write", &err));
    OCL_CHECK(err, cl::Kernel krnl_stencil_SLR0(program, "stencil_SLR0", &err));
    OCL_CHECK(err, cl::Kernel krnl_stencil_SLR1(program, "stencil_SLR1", &err));
    OCL_CHECK(err, cl::Kernel krnl_stencil_SLR2(program, "stencil_SLR2", &err));

    auto end_p = std::chrono::high_resolution_clock::now();
    auto dur_p = end_p -start_p;
    printf("time to program FPGA is %f\n", dur_p.count());


    //Allocate Buffer in Global Memory
    OCL_CHECK(err,cl::Buffer buffer_input0(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, data_size_bytes_div8, grid_u1_d8[0], &err));
    OCL_CHECK(err,cl::Buffer buffer_input1(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, data_size_bytes_div8, grid_u1_d8[1], &err));
    OCL_CHECK(err,cl::Buffer buffer_input2(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, data_size_bytes_div8, grid_u1_d8[2], &err));
    OCL_CHECK(err,cl::Buffer buffer_input3(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, data_size_bytes_div8, grid_u1_d8[3], &err));
    OCL_CHECK(err,cl::Buffer buffer_input4(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, data_size_bytes_div8, grid_u1_d8[4], &err));
    OCL_CHECK(err,cl::Buffer buffer_input5(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, data_size_bytes_div8, grid_u1_d8[5], &err));
    OCL_CHECK(err,cl::Buffer buffer_input6(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, data_size_bytes_div8, grid_u1_d8[6], &err));
    OCL_CHECK(err,cl::Buffer buffer_input7(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, data_size_bytes_div8, grid_u1_d8[7], &err));

    OCL_CHECK(err, cl::Buffer buffer_output0(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, data_size_bytes_div8, grid_u2_d8[0], &err));
    OCL_CHECK(err, cl::Buffer buffer_output1(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, data_size_bytes_div8, grid_u2_d8[1], &err));
    OCL_CHECK(err, cl::Buffer buffer_output2(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, data_size_bytes_div8, grid_u2_d8[2], &err));
    OCL_CHECK(err, cl::Buffer buffer_output3(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, data_size_bytes_div8, grid_u2_d8[3], &err));
    OCL_CHECK(err, cl::Buffer buffer_output4(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, data_size_bytes_div8, grid_u2_d8[4], &err));
    OCL_CHECK(err, cl::Buffer buffer_output5(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, data_size_bytes_div8, grid_u2_d8[5], &err));
    OCL_CHECK(err, cl::Buffer buffer_output6(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, data_size_bytes_div8, grid_u2_d8[6], &err));
    OCL_CHECK(err, cl::Buffer buffer_output7(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, data_size_bytes_div8, grid_u2_d8[7], &err));


    OCL_CHECK(err, cl::Buffer buffer_tile(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, tile_count*sizeof(unsigned int), tile, &err));


    int read_write_offset = data_size_bytes/64;
    //Set the Kernel Arguments
    int narg = 0;
//    tilex_count = 1;
//    tiley_count = 1;
    OCL_CHECK(err, err = krnl_stencil_Read_Write.setArg(narg++, buffer_input0));
    OCL_CHECK(err, err = krnl_stencil_Read_Write.setArg(narg++, buffer_output0));
    OCL_CHECK(err, err = krnl_stencil_Read_Write.setArg(narg++, buffer_input1));
    OCL_CHECK(err, err = krnl_stencil_Read_Write.setArg(narg++, buffer_output1));
    OCL_CHECK(err, err = krnl_stencil_Read_Write.setArg(narg++, buffer_input2));
    OCL_CHECK(err, err = krnl_stencil_Read_Write.setArg(narg++, buffer_output2));
    OCL_CHECK(err, err = krnl_stencil_Read_Write.setArg(narg++, buffer_input3));
    OCL_CHECK(err, err = krnl_stencil_Read_Write.setArg(narg++, buffer_output3));
    OCL_CHECK(err, err = krnl_stencil_Read_Write.setArg(narg++, buffer_input4));
    OCL_CHECK(err, err = krnl_stencil_Read_Write.setArg(narg++, buffer_output4));
    OCL_CHECK(err, err = krnl_stencil_Read_Write.setArg(narg++, buffer_input5));
    OCL_CHECK(err, err = krnl_stencil_Read_Write.setArg(narg++, buffer_output5));
    OCL_CHECK(err, err = krnl_stencil_Read_Write.setArg(narg++, buffer_input6));
    OCL_CHECK(err, err = krnl_stencil_Read_Write.setArg(narg++, buffer_output6));
    OCL_CHECK(err, err = krnl_stencil_Read_Write.setArg(narg++, buffer_input7));
    OCL_CHECK(err, err = krnl_stencil_Read_Write.setArg(narg++, buffer_output7));

    OCL_CHECK(err, err = krnl_stencil_Read_Write.setArg(narg++, buffer_tile));
    OCL_CHECK(err, err = krnl_stencil_Read_Write.setArg(narg++, tilex_count));
    OCL_CHECK(err, err = krnl_stencil_Read_Write.setArg(narg++, tiley_count));
    OCL_CHECK(err, err = krnl_stencil_Read_Write.setArg(narg++, logical_size_x));
    OCL_CHECK(err, err = krnl_stencil_Read_Write.setArg(narg++, logical_size_y));
    OCL_CHECK(err, err = krnl_stencil_Read_Write.setArg(narg++, logical_size_z));
    OCL_CHECK(err, err = krnl_stencil_Read_Write.setArg(narg++, grid_size_x));
    OCL_CHECK(err, err = krnl_stencil_Read_Write.setArg(narg++, n_iter));

    narg = 0;
    OCL_CHECK(err, err = krnl_stencil_SLR0.setArg(narg++, tilex_count));
    OCL_CHECK(err, err = krnl_stencil_SLR0.setArg(narg++, tiley_count));
    OCL_CHECK(err, err = krnl_stencil_SLR0.setArg(narg++, logical_size_x));
    OCL_CHECK(err, err = krnl_stencil_SLR0.setArg(narg++, logical_size_y));
    OCL_CHECK(err, err = krnl_stencil_SLR0.setArg(narg++, logical_size_z));
    OCL_CHECK(err, err = krnl_stencil_SLR0.setArg(narg++, grid_size_x));
    OCL_CHECK(err, err = krnl_stencil_SLR0.setArg(narg++, n_iter));

    narg = 0;
    OCL_CHECK(err, err = krnl_stencil_SLR1.setArg(narg++, tilex_count));
    OCL_CHECK(err, err = krnl_stencil_SLR1.setArg(narg++, tiley_count));
    OCL_CHECK(err, err = krnl_stencil_SLR1.setArg(narg++, logical_size_x));
    OCL_CHECK(err, err = krnl_stencil_SLR1.setArg(narg++, logical_size_y));
    OCL_CHECK(err, err = krnl_stencil_SLR1.setArg(narg++, logical_size_z));
    OCL_CHECK(err, err = krnl_stencil_SLR1.setArg(narg++, grid_size_x));
    OCL_CHECK(err, err = krnl_stencil_SLR1.setArg(narg++, n_iter));

    narg = 0;
    OCL_CHECK(err, err = krnl_stencil_SLR2.setArg(narg++, tilex_count));
    OCL_CHECK(err, err = krnl_stencil_SLR2.setArg(narg++, tiley_count));
    OCL_CHECK(err, err = krnl_stencil_SLR2.setArg(narg++, logical_size_x));
    OCL_CHECK(err, err = krnl_stencil_SLR2.setArg(narg++, logical_size_y));
    OCL_CHECK(err, err = krnl_stencil_SLR2.setArg(narg++, logical_size_z));
    OCL_CHECK(err, err = krnl_stencil_SLR2.setArg(narg++, grid_size_x));
    OCL_CHECK(err, err = krnl_stencil_SLR2.setArg(narg++, n_iter));


    OCL_CHECK(err,
              err = q.enqueueMigrateMemObjects({buffer_input0, buffer_input1, buffer_input2, buffer_input3,
    										buffer_input4, buffer_input5, buffer_input6, buffer_input7, buffer_tile},
                                               0 /* 0 means from host*/));

    uint64_t nsduration, startns, endns;
    q.finish();
    auto start = std::chrono::high_resolution_clock::now();





  //  tile[0] = (act_sizex-tilex_size) | tilex_size << 16;
//    tilex_count = 2;

//    tile[0] = (act_sizex-tilex_size) | tilex_size << 16;
//    tile[1] = (act_sizex-tilex_size) | tilex_size << 16;
//        OCL_CHECK(err,
//                  err = q.enqueueMigrateMemObjects({buffer_tile},
//                                                   0 /* 0 means from host*/));

    cl::Event event;
	OCL_CHECK(err, err = q.enqueueTask(krnl_stencil_Read_Write, NULL, &event));
	OCL_CHECK(err, err = q.enqueueTask(krnl_stencil_SLR0));
	OCL_CHECK(err, err = q.enqueueTask(krnl_stencil_SLR1));
	OCL_CHECK(err, err = q.enqueueTask(krnl_stencil_SLR2));

	OCL_CHECK(err, err=event.wait());
	endns = OCL_CHECK(err, event.getProfilingInfo<CL_PROFILING_COMMAND_END>(&err));
	startns = OCL_CHECK(err, event.getProfilingInfo<CL_PROFILING_COMMAND_START>(&err));
	nsduration = endns - startns;
	double k_time = nsduration/(1000000000.0);
	q.finish();


//	tile[0] = (act_sizex-tilex_size) | tilex_size << 16;
//    OCL_CHECK(err,
//              err = q.enqueueMigrateMemObjects({buffer_tile},
//                                               0 /* 0 means from host*/));
//	OCL_CHECK(err, err = q.enqueueTask(krnl_stencil_Read_Write, NULL, &event));
//	OCL_CHECK(err, err = q.enqueueTask(krnl_stencil_SLR0));
//	OCL_CHECK(err, err = q.enqueueTask(krnl_stencil_SLR1));
//	OCL_CHECK(err, err = q.enqueueTask(krnl_stencil_SLR2));
//	q.finish();






//    char fine_name[100];
//    for(unsigned short shift =0; shift <= 0; shift += 16){
//    	sprintf(fine_name, "4K_cross_1RW_HBM.csv");
//    	FILE* fptr = fopen(fine_name, "w");
//
//
//		fprintf(fptr, "Transfer Size(Bytes), Stride(Bytes), Number of Transfers,  Total_Bandwidth(GB/s)\n");
//		//Launch the Kernel
//		for(int i = 16; i <= 8192; i = i *2){
//			for(unsigned short tilex = 16; tilex <= 2048 && tilex <= i; tilex = tilex + 16){
//
//				offset_x = 0; //((1024-128)/16)*16;
//				tile[0] = offset_x | (tilex << 16);
//				tile[0+tilex_count] = 0* effective_tiley_size  | (tiley_size << 16);
//				OCL_CHECK(err,
//						  err = q.enqueueMigrateMemObjects({buffer_tile},
//														   0 /* 0 means from host*/));
//				q.finish();
//				cl::Event event;
//				OCL_CHECK(err, err = krnl_Read_Write.setArg(5, i-2));
//				OCL_CHECK(err, err = krnl_Read_Write.setArg(8, i));
//				OCL_CHECK(err, err = q.enqueueTask(krnl_Read_Write, NULL, &event));
//				OCL_CHECK(err, err=event.wait());
//				endns = OCL_CHECK(err, event.getProfilingInfo<CL_PROFILING_COMMAND_END>(&err));
//				startns = OCL_CHECK(err, event.getProfilingInfo<CL_PROFILING_COMMAND_START>(&err));
//				nsduration = endns - startns;
//				q.finish();
//
//
//				double k_time = nsduration/(1000000000.0);
//				total_plane_sizeR = tilex * tiley_size;
//				float logic_bandwidthR = (total_plane_sizeR * act_sizez * sizeof(float) * 2.0 * n_iter)/(k_time * 1000.0 * 1000 * 1000);
//				fprintf(fptr, "%d, %d, %d, %f\n", tilex*4, i*4, tiley_size*act_sizez*n_iter*4, 2*logic_bandwidthR);
//				printf("tilex:%d i:%d\n", tilex,i);
//				printf("%d, %d, %d, %f\n", tilex*4, i*4, tiley_size*act_sizez*n_iter*4, 2*logic_bandwidthR);
//			}
//			fprintf(fptr, ",,,\n");
//			fprintf(fptr, ",,,\n");
//		}
//
//
//		fclose(fptr);
//	}


    auto finish = std::chrono::high_resolution_clock::now();
    OCL_CHECK(err,
              err = q.enqueueMigrateMemObjects({buffer_input0, buffer_input1, buffer_input2, buffer_input3, buffer_input4, buffer_input5, buffer_input6, buffer_input7},
                                       	   	   	   CL_MIGRATE_MEM_OBJECT_HOST));

    OCL_CHECK(err,
                  err = q.enqueueMigrateMemObjects({buffer_output0, buffer_output1, buffer_output2, buffer_output3, buffer_output4, buffer_output5, buffer_output6, buffer_output7},
                                                   CL_MIGRATE_MEM_OBJECT_HOST));

    q.finish();

    merge_grid(grid_u1_d8, grid_u1_d, act_sizex, act_sizey, act_sizez, grid_size_x, grid_size_y, grid_size_z);
    merge_grid(grid_u2_d8, grid_u2_d, act_sizex, act_sizey, act_sizez, grid_size_x, grid_size_y, grid_size_z);

  for(int itr = 0; itr < 3*n_iter; itr++){
      stencil_computation(grid_u1, grid_u2, act_sizex, act_sizey, act_sizez, grid_size_x, grid_size_y, grid_size_z);
      stencil_computation(grid_u2, grid_u1, act_sizex, act_sizey, act_sizez, grid_size_x, grid_size_y, grid_size_z);
  }
    
    std::chrono::duration<double> elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(finish - start);



  printf("Profiling time is %f\n", k_time);

  printf("Runtime on FPGA is %f seconds\n", k_time);
  double error = square_error(grid_u1, grid_u1_d, act_sizex, act_sizey, act_sizez, grid_size_x, grid_size_y, grid_size_x);
  float bandwidth = (logical_size_x * logical_size_y * logical_size_z * sizeof(float) * 4.0 * n_iter*3)/(k_time * 1000.0 * 1000 * 1000);
  float logic_bandwidthR = (total_plane_sizeR * act_sizez * sizeof(float) * 2.0 * n_iter*3)/(k_time * 1000.0 * 1000 * 1000);
  float logic_bandwidthW = (total_plane_sizeW * act_sizez * sizeof(float) * 2.0 * n_iter*3)/(k_time * 1000.0 * 1000 * 1000);
  printf("\nMean Square error is  %f\n\n", error/(logical_size_x * logical_size_y));
  printf("grid_size_x:%d grid_size_y:%d grid_size_z:%d, offset_x:%d\n",grid_size_x,grid_size_y,grid_size_z, offset_x);
  printf("\nBandwidth is %f %f %f %f\n", bandwidth, logic_bandwidthR, logic_bandwidthW, 2*logic_bandwidthW);








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
