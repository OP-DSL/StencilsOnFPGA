
// standard headers
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "xcl2.hpp"
#include <chrono>
#include "stencil_cpu.h"



/******************************************************************************
* Main program
*******************************************************************************/
int main(int argc, char **argv)
{

  //Mesh
  struct Grid_Parameter data_g;
  data_g.logical_size_x = 20;
  data_g.logical_size_y = 20;
  data_g.logical_size_z = 20;
  data_g.batch = 1;

  int n_iter = 10;


  const char* pch;
  for ( int n = 1; n < argc; n++ ) {
    pch = strstr(argv[n], "-sizex=");
    if(pch != NULL) {
      data_g.logical_size_x = atoi ( argv[n] + 7 ); continue;
    }
    pch = strstr(argv[n], "-sizey=");
    if(pch != NULL) {
      data_g.logical_size_y = atoi ( argv[n] + 7 ); continue;
    }
    pch = strstr(argv[n], "-sizez=");
	if(pch != NULL) {
	  data_g.logical_size_z = atoi ( argv[n] + 7 ); continue;
	}
    pch = strstr(argv[n], "-iters=");
    if(pch != NULL) {
      n_iter = atoi ( argv[n] + 7 ); continue;
    }
    pch = strstr(argv[n], "-batch=");
	if(pch != NULL) {
	  data_g.batch = atoi ( argv[n] + 7 ); continue;
	}
  }


  printf("Grid: %dx%dx%d , %d iterations, %d batches\n",data_g.logical_size_x, data_g.logical_size_y, data_g.logical_size_z, n_iter,  data_g.batch);

  data_g.act_size_x = data_g.logical_size_x + 2;
  data_g.act_size_y = data_g.logical_size_y + 2;
  data_g.act_size_z = data_g.logical_size_z + 2;


  data_g.grid_size_x = (data_g.act_size_x % 8) != 0 ? (data_g.act_size_x/8 +1) * 8 : data_g.act_size_x;
  data_g.grid_size_y = data_g.act_size_y;
  data_g.grid_size_z = data_g.act_size_z;




//  unsigned short effective_tile_size= tile_size - 0;
  unsigned int data_size_bytes = data_g.grid_size_x * data_g.grid_size_y * data_g.grid_size_z *sizeof(float)*data_g.batch;
  data_size_bytes = data_size_bytes % 16 != 0 ? (data_size_bytes/16 + 1)*16 : data_size_bytes;
  float * grid_u1 = (float*)aligned_alloc(4096, data_size_bytes);
  float * grid_u2 = (float*)aligned_alloc(4096, data_size_bytes);

  float * grid_u1_d = (float*)aligned_alloc(4096, data_size_bytes);
  float * grid_u2_d = (float*)aligned_alloc(4096, data_size_bytes);


  initialise_grid(grid_u1, data_g);
  copy_grid(grid_u1, grid_u1_d, data_size_bytes);

  //OPENCL HOST CODE AREA START

    auto binaryFile = argv[1];
    cl_int err;
//    cl::Event event;

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
    std::chrono::duration<double> dur_p = end_p -start_p;
    printf("time to program FPGA is %f\n", dur_p.count());


    //Allocate Buffer in Global Memory
    OCL_CHECK(err, cl::Buffer buffer_input(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, data_size_bytes, grid_u1_d, &err));
    OCL_CHECK(err, cl::Buffer buffer_output(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, data_size_bytes, grid_u2_d, &err));



    //Set the Kernel Arguments
    int narg = 0;
    OCL_CHECK(err, err = krnl_Read_Write.setArg(narg++, buffer_input));
    OCL_CHECK(err, err = krnl_Read_Write.setArg(narg++, buffer_output));
    OCL_CHECK(err, err = krnl_Read_Write.setArg(narg++, data_g.logical_size_x));
    OCL_CHECK(err, err = krnl_Read_Write.setArg(narg++, data_g.logical_size_y));
    OCL_CHECK(err, err = krnl_Read_Write.setArg(narg++, data_g.logical_size_z));
    OCL_CHECK(err, err = krnl_Read_Write.setArg(narg++, data_g.grid_size_x));
    OCL_CHECK(err, err = krnl_Read_Write.setArg(narg++, data_g.batch));
    OCL_CHECK(err, err = krnl_Read_Write.setArg(narg++, n_iter));

    narg = 0;
    OCL_CHECK(err, err = krnl_slr0.setArg(narg++, data_g.logical_size_x));
    OCL_CHECK(err, err = krnl_slr0.setArg(narg++, data_g.logical_size_y));
    OCL_CHECK(err, err = krnl_slr0.setArg(narg++, data_g.logical_size_z));
    OCL_CHECK(err, err = krnl_slr0.setArg(narg++, data_g.grid_size_x));
    OCL_CHECK(err, err = krnl_slr0.setArg(narg++, data_g.batch));
    OCL_CHECK(err, err = krnl_slr0.setArg(narg++, n_iter));

    narg = 0;
    OCL_CHECK(err, err = krnl_slr1.setArg(narg++, data_g.logical_size_x));
    OCL_CHECK(err, err = krnl_slr1.setArg(narg++, data_g.logical_size_y));
    OCL_CHECK(err, err = krnl_slr1.setArg(narg++, data_g.logical_size_z));
    OCL_CHECK(err, err = krnl_slr1.setArg(narg++, data_g.grid_size_x));
    OCL_CHECK(err, err = krnl_slr1.setArg(narg++, data_g.batch));
    OCL_CHECK(err, err = krnl_slr1.setArg(narg++, n_iter));

  	narg = 0;
    OCL_CHECK(err, err = krnl_slr2.setArg(narg++, data_g.logical_size_x));
    OCL_CHECK(err, err = krnl_slr2.setArg(narg++, data_g.logical_size_y));
    OCL_CHECK(err, err = krnl_slr2.setArg(narg++, data_g.logical_size_z));
  	OCL_CHECK(err, err = krnl_slr2.setArg(narg++, data_g.grid_size_x));
  	OCL_CHECK(err, err = krnl_slr2.setArg(narg++, data_g.batch));
  	OCL_CHECK(err, err = krnl_slr2.setArg(narg++, n_iter));

    //Copy input data to device global memory
    OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_input}, 0 /* 0 means from host*/));


	//Launch the Kernel
    cl::Event event;
    OCL_CHECK(err, err = q.enqueueTask(krnl_Read_Write, NULL, &event));
	OCL_CHECK(err, err = q.enqueueTask(krnl_slr0));
	OCL_CHECK(err, err = q.enqueueTask(krnl_slr1));
	OCL_CHECK(err, err = q.enqueueTask(krnl_slr2));

	OCL_CHECK(err, err=event.wait());
	uint64_t endns = OCL_CHECK(err, event.getProfilingInfo<CL_PROFILING_COMMAND_END>(&err));
	uint64_t startns = OCL_CHECK(err, event.getProfilingInfo<CL_PROFILING_COMMAND_START>(&err));
	uint64_t nsduration = endns - startns;
	double k_time = nsduration/(1000000000.0);
	q.finish();



    auto finish = std::chrono::high_resolution_clock::now();
    OCL_CHECK(err,
              err = q.enqueueMigrateMemObjects({buffer_input},
                                               CL_MIGRATE_MEM_OBJECT_HOST));

    OCL_CHECK(err,
                  err = q.enqueueMigrateMemObjects({buffer_output},
                                                   CL_MIGRATE_MEM_OBJECT_HOST));

    q.finish();

  for(int itr = 0; itr < n_iter*27; itr++){
      stencil_computation(grid_u1, grid_u2, data_g);
      stencil_computation(grid_u2, grid_u1, data_g);
  }
    

  printf("Runtime on FPGA is %f seconds\n", k_time);
  double error = square_error(grid_u1, grid_u1_d, data_g);
  float bandwidth = (data_g.logical_size_x * data_g.logical_size_y * data_g.logical_size_z * sizeof(float) * 4.0 * n_iter * 3* data_g.batch)/(k_time * 1000 * 1000 * 1000);
  printf("\nMean Square error is  %f\n\n", error/(data_g.logical_size_x * data_g.logical_size_y));
  printf("\nBandwidth is %f\n", bandwidth);

  // Releasing allocated memory
  free(grid_u1);
  free(grid_u2);
  free(grid_u1_d);
  free(grid_u2_d);

  return 0;
}
