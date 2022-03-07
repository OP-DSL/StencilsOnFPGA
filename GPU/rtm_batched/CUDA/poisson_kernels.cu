//
// auto-generated by ops.py//

// header
#define OPS_API 2
#define OPS_3D
#include "ops_lib_cpp.h"

#include "ops_cuda_reduction.h"
#include "ops_cuda_rt_support.h"

#include <cuComplex.h>

#ifdef OPS_MPI
#include "ops_mpi_core.h"
#endif
// global constants
__constant__ float dx;
__constant__ float dy;
__constant__ float dz;
__constant__ int nx;
__constant__ int ny;
__constant__ int nz;
__constant__ int pml_width;
__constant__ int half;
__constant__ int order;

void ops_init_backend() {}

void ops_decl_const_char(int dim, char const *type, int size, char *dat,
                         char const *name) {
  if (!strcmp(name, "dx")) {
    cutilSafeCall(cudaMemcpyToSymbol(dx, dat, dim * size));
  } else if (!strcmp(name, "dy")) {
    cutilSafeCall(cudaMemcpyToSymbol(dy, dat, dim * size));
  } else if (!strcmp(name, "dz")) {
    cutilSafeCall(cudaMemcpyToSymbol(dz, dat, dim * size));
  } else if (!strcmp(name, "nx")) {
    cutilSafeCall(cudaMemcpyToSymbol(nx, dat, dim * size));
  } else if (!strcmp(name, "ny")) {
    cutilSafeCall(cudaMemcpyToSymbol(ny, dat, dim * size));
  } else if (!strcmp(name, "nz")) {
    cutilSafeCall(cudaMemcpyToSymbol(nz, dat, dim * size));
  } else if (!strcmp(name, "pml_width")) {
    cutilSafeCall(cudaMemcpyToSymbol(pml_width, dat, dim * size));
  } else if (!strcmp(name, "half")) {
    cutilSafeCall(cudaMemcpyToSymbol(half, dat, dim * size));
  } else if (!strcmp(name, "order")) {
    cutilSafeCall(cudaMemcpyToSymbol(order, dat, dim * size));
  } else {
    printf("error: unknown const name\n");
    exit(1);
  }
}

// user kernel files
#include "fd3d_pml_kernel1_cuda_kernel.cu"
#include "fd3d_pml_kernel2_cuda_kernel.cu"
#include "fd3d_pml_kernel3_cuda_kernel.cu"
#include "rtm_kernel_populate_cuda_kernel.cu"
