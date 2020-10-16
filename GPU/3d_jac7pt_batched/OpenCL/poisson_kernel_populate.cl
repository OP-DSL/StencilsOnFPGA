//
// auto-generated by ops.py
//

#ifdef OCL_FMA
#pragma OPENCL FP_CONTRACT ON
#else
#pragma OPENCL FP_CONTRACT OFF
#endif
#pragma OPENCL EXTENSION cl_khr_fp64:enable

#include "user_types.h"
#define OPS_3D
#define OPS_API 2
#define OPS_NO_GLOBALS
#include "ops_macros.h"
#include "ops_opencl_reduction.h"

#ifndef MIN
#define MIN(a,b) ((a<b) ? (a) : (b))
#endif
#ifndef MAX
#define MAX(a,b) ((a>b) ? (a) : (b))
#endif
#ifndef SIGN
#define SIGN(a,b) ((b<0.0) ? (a*(-1)) : (a))
#endif
#define OPS_READ 0
#define OPS_WRITE 1
#define OPS_RW 2
#define OPS_INC 3
#define OPS_MIN 4
#define OPS_MAX 5
#define ZERO_double 0.0;
#define INFINITY_double INFINITY;
#define ZERO_float 0.0f;
#define INFINITY_float INFINITY;
#define ZERO_int 0;
#define INFINITY_int INFINITY;
#define ZERO_uint 0;
#define INFINITY_uint INFINITY;
#define ZERO_ll 0;
#define INFINITY_ll INFINITY;
#define ZERO_ull 0;
#define INFINITY_ull INFINITY;
#define ZERO_bool 0;

//user function

void poisson_kernel_populate(const int *idx,
  ptr_float u,
  ptr_float f,
  ptr_float ref, const float dx, const float dy)
{
  float x = dx * (float)(idx[0]);
  float y = dy * (float)(idx[1]);

  OPS_ACCS(u, 0,0,0) = myfun(sin(M_PI*x),cos(2.0*M_PI*y))-1.0;
  OPS_ACCS(f, 0,0,0) = -5.0*M_PI*M_PI*sin(M_PI*x)*cos(2.0*M_PI*y);
  OPS_ACCS(ref, 0,0,0) = sin(M_PI*x)*cos(2.0*M_PI*y);

}


__kernel void ops_poisson_kernel_populate(
__global float* restrict arg1,
__global float* restrict arg2,
__global float* restrict arg3,
const float dx,
const float dy,
const int base1,
const int base2,
const int base3,
int arg_idx0, int arg_idx1, int arg_idx2,
const int size0,
const int size1,
const int size2 ){


  int idx_y = get_global_id(1);
  int idx_z = get_global_id(2);
  int idx_x = get_global_id(0);

  int arg_idx[3];
  arg_idx[0] = arg_idx0+idx_x;
  arg_idx[1] = arg_idx1+idx_y;
  arg_idx[2] = arg_idx2+idx_z;
  if (idx_x < size0 && idx_y < size1 && idx_z < size2) {
    ptr_float ptr1 = { &arg1[base1 + idx_x * 1*1 + idx_y * 1*1 * xdim1_poisson_kernel_populate + idx_z * 1*1 * xdim1_poisson_kernel_populate * ydim1_poisson_kernel_populate], xdim1_poisson_kernel_populate, ydim1_poisson_kernel_populate};
    ptr_float ptr2 = { &arg2[base2 + idx_x * 1*1 + idx_y * 1*1 * xdim2_poisson_kernel_populate + idx_z * 1*1 * xdim2_poisson_kernel_populate * ydim2_poisson_kernel_populate], xdim2_poisson_kernel_populate, ydim2_poisson_kernel_populate};
    ptr_float ptr3 = { &arg3[base3 + idx_x * 1*1 + idx_y * 1*1 * xdim3_poisson_kernel_populate + idx_z * 1*1 * xdim3_poisson_kernel_populate * ydim3_poisson_kernel_populate], xdim3_poisson_kernel_populate, ydim3_poisson_kernel_populate};
    poisson_kernel_populate(arg_idx,
                   ptr1,
                   ptr2,
                   ptr3,
                   dx,
                   dy);
  }

}
