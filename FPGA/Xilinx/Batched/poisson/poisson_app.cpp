
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

  //setting Mesh default parameters
  struct Grid_Parameter data_g;
  data_g.logical_size_x = 20;
  data_g.logical_size_y = 20;
  data_g.batch =2;

  // number of iterations
  int n_iter = 10;

  // setting grid parameters given by user
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
    pch = strstr(argv[n], "-iters=");
    if(pch != NULL) {
      n_iter = atoi ( argv[n] + 7 ); continue;
    }

    pch = strstr(argv[n], "-batch=");
	if(pch != NULL) {
	  data_g.batch = atoi ( argv[n] + 7 ); continue;
	}
  }

  printf("Grid: %dx%d , %d iterations, %d batches\n", data_g.logical_size_x, data_g.logical_size_y, n_iter, data_g.batch);

  // adding boundary
  data_g.act_size_x = data_g.logical_size_x + 2;
  data_g.act_size_y = data_g.logical_size_y + 2;

  // padding each row such that it is multiple of Vectorisation factor
  data_g.grid_size_x = (data_g.act_size_x % 8) != 0 ? (data_g.act_size_x/8 +1) * 8 : data_g.act_size_x;
  data_g.grid_size_y = data_g.act_size_y;
  
  // allocating memory for host program and FPGA buffers
  unsigned int data_size_bytes = data_g.grid_size_x * data_g.grid_size_y * sizeof(float)*data_g.batch;
  data_size_bytes = (data_size_bytes % 16 != 0) ? (data_size_bytes/16 +1)*16 : data_size_bytes;
  if(data_size_bytes >= 4000000000){
	  printf("Maximum buffer size is exceeded!\n");
	  return 0;
  }
  float * grid_u1 = (float*)aligned_alloc(4096, data_size_bytes);
  float * grid_u2 = (float*)aligned_alloc(4096, data_size_bytes);

  float * grid_u1_d = (float*)aligned_alloc(4096, data_size_bytes);
  float * grid_u2_d = (float*)aligned_alloc(4096, data_size_bytes);

  // setting boundary value and copying to FPGA buffer
  initialise_grid(grid_u1, data_g);
  copy_grid(grid_u1, grid_u1_d, data_g);



  //OPENCL HOST CODE AREA START
    auto binaryFile = argv[1];
    cl_int err;
    cl::Event event;

    auto devices = xcl::get_xil_devices();
    auto device = devices[0];

    OCL_CHECK(err, cl::Context context(device, NULL, NULL, NULL, &err));
    OCL_CHECK(err, cl::CommandQueue q(context, device, CL_QUEUE_PROFILING_ENABLE | CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, &err));
    OCL_CHECK(err, std::string device_name = device.getInfo<CL_DEVICE_NAME>(&err));


    //Create Program and Kernel
    auto fileBuf = xcl::read_binary_file(binaryFile);
    cl::Program::Binaries bins{{fileBuf.data(), fileBuf.size()}};
    devices.resize(1);
    OCL_CHECK(err, cl::Program program(context, devices, bins, NULL, &err));
    OCL_CHECK(err, cl::Kernel krnl_slr0(program, "stencil_SLR0", &err));
    OCL_CHECK(err, cl::Kernel krnl_slr1(program, "stencil_SLR1", &err));
    OCL_CHECK(err, cl::Kernel krnl_slr2(program, "stencil_SLR2", &err));



    //Allocate Buffer in Global Memory
    OCL_CHECK(err, cl::Buffer buffer_input(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, data_size_bytes, grid_u1_d, &err));
    OCL_CHECK(err, cl::Buffer buffer_output(context, CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY, data_size_bytes, grid_u2_d, &err));


    //Set the Kernel Arguments
    int narg = 0;
    OCL_CHECK(err, err = krnl_slr0.setArg(narg++, buffer_input));
    OCL_CHECK(err, err = krnl_slr0.setArg(narg++, buffer_output));
    OCL_CHECK(err, err = krnl_slr0.setArg(narg++, data_g.logical_size_x));
    OCL_CHECK(err, err = krnl_slr0.setArg(narg++, data_g.logical_size_y));
    OCL_CHECK(err, err = krnl_slr0.setArg(narg++, data_g.grid_size_x));
    OCL_CHECK(err, err = krnl_slr0.setArg(narg++, n_iter));
    OCL_CHECK(err, err = krnl_slr0.setArg(narg++, data_g.batch));

    narg = 0;
	OCL_CHECK(err, err = krnl_slr1.setArg(narg++, data_g.logical_size_x));
	OCL_CHECK(err, err = krnl_slr1.setArg(narg++, data_g.logical_size_y));
	OCL_CHECK(err, err = krnl_slr1.setArg(narg++, data_g.grid_size_x));
	OCL_CHECK(err, err = krnl_slr1.setArg(narg++, n_iter));
	OCL_CHECK(err, err = krnl_slr1.setArg(narg++, data_g.batch));

	narg = 0;
	OCL_CHECK(err, err = krnl_slr2.setArg(narg++, data_g.logical_size_x));
	OCL_CHECK(err, err = krnl_slr2.setArg(narg++, data_g.logical_size_y));
	OCL_CHECK(err, err = krnl_slr2.setArg(narg++, data_g.grid_size_x));
	OCL_CHECK(err, err = krnl_slr2.setArg(narg++, n_iter));
	OCL_CHECK(err, err = krnl_slr2.setArg(narg++, data_g.batch));

    //Copy input data to device global memory
    OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_input}, 0 /* 0 means from host*/));

    uint64_t wtime = 0;
    uint64_t nstimestart, nstimeend;
    auto start = std::chrono::high_resolution_clock::now();




	//Launch the Kernel
	OCL_CHECK(err, err = q.enqueueTask(krnl_slr0));
	OCL_CHECK(err, err = q.enqueueTask(krnl_slr1));
	OCL_CHECK(err, err = q.enqueueTask(krnl_slr2));
	q.finish();


    auto finish = std::chrono::high_resolution_clock::now();

    //Copy Result from Device Global Memory to Host Local Memory
    OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_input}, CL_MIGRATE_MEM_OBJECT_HOST));

    q.finish();


  // golden stencil computation on host
  for(int itr = 0; itr < n_iter*60; itr++){
      stencil_computation(grid_u1, grid_u2, data_g);
      stencil_computation(grid_u2, grid_u1, data_g);
  }
    
  std::chrono::duration<double> elapsed = finish - start;

  printf("Runtime on FPGA is %f seconds\n", elapsed.count());
  double error = square_error(grid_u1, grid_u1_d, data_g);
  float bandwidth = (data_g.logical_size_x * data_g.logical_size_y * sizeof(float) * 4.0 * n_iter * data_g.batch)/(elapsed.count() * 1000 * 1000 * 1000);
  printf("\nMean Square error is  %f\n\n", error/(data_g.logical_size_x * data_g.logical_size_y));
  printf("\nOPS Bandwidth is %f\n", bandwidth);

  free(grid_u1);
  free(grid_u2);
  free(grid_u1_d);
  free(grid_u2_d);

  return 0;
}
