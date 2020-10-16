//
// auto-generated by ops.py//

//header
#define OPS_API 2
#define OPS_3D
#include "stdlib.h"
#include "stdio.h"
#include "ops_lib_cpp.h"
#include "ops_opencl_rt_support.h"
#include "user_types.h"
#ifdef OPS_MPI
#include "ops_mpi_core.h"
#endif
//global constants
extern float dx;
extern float dy;
extern float dz;


void ops_init_backend() {}

//this needs to be a platform specific copy symbol to device function
void ops_decl_const_char( int dim, char const * type, int typeSize, char * dat, char const * name ) {
  cl_int ret = 0;
  if (OPS_instance::getOPSInstance()->opencl_instance->OPS_opencl_core.constant == NULL) {
    OPS_instance::getOPSInstance()->opencl_instance->OPS_opencl_core.constant = (cl_mem*) malloc((3)*sizeof(cl_mem));
    for ( int i=0; i<3; i++ ){
      OPS_instance::getOPSInstance()->opencl_instance->OPS_opencl_core.constant[i] = NULL;
    }
  }
  if (!strcmp(name,"dx")) {
    if (OPS_instance::getOPSInstance()->opencl_instance->OPS_opencl_core.constant[0] == NULL) {
      OPS_instance::getOPSInstance()->opencl_instance->OPS_opencl_core.constant[0] = clCreateBuffer(OPS_instance::getOPSInstance()->opencl_instance->OPS_opencl_core.context, CL_MEM_READ_ONLY, dim*typeSize, NULL, &ret);
      clSafeCall( ret );
    }
    //Write the new constant to the memory of the device
    clSafeCall( clEnqueueWriteBuffer(OPS_instance::getOPSInstance()->opencl_instance->OPS_opencl_core.command_queue, OPS_instance::getOPSInstance()->opencl_instance->OPS_opencl_core.constant[0], CL_TRUE, 0, dim*typeSize, (void*) dat, 0, NULL, NULL) );
    clSafeCall( clFlush(OPS_instance::getOPSInstance()->opencl_instance->OPS_opencl_core.command_queue) );
    clSafeCall( clFinish(OPS_instance::getOPSInstance()->opencl_instance->OPS_opencl_core.command_queue) );
  }
  else
  if (!strcmp(name,"dy")) {
    if (OPS_instance::getOPSInstance()->opencl_instance->OPS_opencl_core.constant[1] == NULL) {
      OPS_instance::getOPSInstance()->opencl_instance->OPS_opencl_core.constant[1] = clCreateBuffer(OPS_instance::getOPSInstance()->opencl_instance->OPS_opencl_core.context, CL_MEM_READ_ONLY, dim*typeSize, NULL, &ret);
      clSafeCall( ret );
    }
    //Write the new constant to the memory of the device
    clSafeCall( clEnqueueWriteBuffer(OPS_instance::getOPSInstance()->opencl_instance->OPS_opencl_core.command_queue, OPS_instance::getOPSInstance()->opencl_instance->OPS_opencl_core.constant[1], CL_TRUE, 0, dim*typeSize, (void*) dat, 0, NULL, NULL) );
    clSafeCall( clFlush(OPS_instance::getOPSInstance()->opencl_instance->OPS_opencl_core.command_queue) );
    clSafeCall( clFinish(OPS_instance::getOPSInstance()->opencl_instance->OPS_opencl_core.command_queue) );
  }
  else
  if (!strcmp(name,"dz")) {
    if (OPS_instance::getOPSInstance()->opencl_instance->OPS_opencl_core.constant[2] == NULL) {
      OPS_instance::getOPSInstance()->opencl_instance->OPS_opencl_core.constant[2] = clCreateBuffer(OPS_instance::getOPSInstance()->opencl_instance->OPS_opencl_core.context, CL_MEM_READ_ONLY, dim*typeSize, NULL, &ret);
      clSafeCall( ret );
    }
    //Write the new constant to the memory of the device
    clSafeCall( clEnqueueWriteBuffer(OPS_instance::getOPSInstance()->opencl_instance->OPS_opencl_core.command_queue, OPS_instance::getOPSInstance()->opencl_instance->OPS_opencl_core.constant[2], CL_TRUE, 0, dim*typeSize, (void*) dat, 0, NULL, NULL) );
    clSafeCall( clFlush(OPS_instance::getOPSInstance()->opencl_instance->OPS_opencl_core.command_queue) );
    clSafeCall( clFinish(OPS_instance::getOPSInstance()->opencl_instance->OPS_opencl_core.command_queue) );
  }
  else
  {
    printf("error: unknown const name\n"); exit(1);
  }
}



void buildOpenCLKernels() {
  static bool isbuilt = false;

  if(!isbuilt) {
    //clSafeCall( clUnloadCompiler() );

    OPS_instance::getOPSInstance()->opencl_instance->OPS_opencl_core.n_kernels = 6;
    OPS_instance::getOPSInstance()->opencl_instance->OPS_opencl_core.kernel = (cl_kernel*) malloc(6*sizeof(cl_kernel));
  }
  isbuilt = true;
}

//user kernel files
#include "poisson_kernel_error_opencl_kernel.cpp"
#include "poisson_kernel_stencil_opencl_kernel.cpp"
#include "poisson_kernel_initialguess_opencl_kernel.cpp"
#include "poisson_kernel_update_opencl_kernel.cpp"
#include "poisson_kernel_populate_opencl_kernel.cpp"
