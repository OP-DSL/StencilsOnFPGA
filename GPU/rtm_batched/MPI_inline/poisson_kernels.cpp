//
// auto-generated by ops.py
//

#include "./MPI_inline/poisson_common.h"


void ops_init_backend() {}

void ops_decl_const_char2(int dim, char const *type,
int size, char *dat, char const *name){
  if (!strcmp(name,"dx")) {
    dx = *(float*)dat;
  }
  else
  if (!strcmp(name,"dy")) {
    dy = *(float*)dat;
  }
  else
  if (!strcmp(name,"dz")) {
    dz = *(float*)dat;
  }
  else
  if (!strcmp(name,"nx")) {
    nx = *(int*)dat;
  }
  else
  if (!strcmp(name,"ny")) {
    ny = *(int*)dat;
  }
  else
  if (!strcmp(name,"nz")) {
    nz = *(int*)dat;
  }
  else
  if (!strcmp(name,"pml_width")) {
    pml_width = *(int*)dat;
  }
  else
  if (!strcmp(name,"half")) {
    half = *(int*)dat;
  }
  else
  if (!strcmp(name,"order")) {
    order = *(int*)dat;
  }
  else
  {
    printf("error: unknown const name\n"); exit(1);
  }
}

//user kernel files
#include "rtm_kernel_populate_mpiinline_kernel.cpp"
#include "fd3d_pml_kernel_mpiinline_kernel.cpp"
#include "calc_ytemp_kernel_mpiinline_kernel.cpp"
#include "calc_ytemp2_kernel_mpiinline_kernel.cpp"
#include "final_update_kernel_mpiinline_kernel.cpp"
