//
// auto-generated by ops.py
//
__constant__ int dims_poisson_kernel_error [3][4];
static int dims_poisson_kernel_error_h [3][4] = {0};

//user function


__global__ void ops_poisson_kernel_error(
float* __restrict u_p,
float* __restrict ref_p,
float* __restrict err_p,
int bounds_0_l, int bounds_0_u, int bounds_1_l, int bounds_1_u,
int bounds_2_l, int bounds_2_u, int bounds_3_l, int bounds_3_u) {

  float err[1];
  err[0] = ZERO_float;


  int n_2 = bounds_2_l + blockDim.z * blockIdx.z + threadIdx.z;
  int n_3 = n_2/(bounds_2_u-bounds_2_l);
  #ifdef OPS_BATCHED
  n_2 = n_2%(bounds_2_u-bounds_2_l); 
  #endif
  int n_1 = bounds_1_l + blockDim.y * blockIdx.y + threadIdx.y;
  int n_0 = bounds_0_l + blockDim.x * blockIdx.x + threadIdx.x;

  if (n_0 < bounds_0_u && n_1 < bounds_1_u && n_2 < bounds_2_u && n_3 < bounds_3_u) {
    const ACC<float> u(dims_poisson_kernel_error[0][0], dims_poisson_kernel_error[0][1], dims_poisson_kernel_error[0][2], u_p + n_0 + n_1 * dims_poisson_kernel_error[0][0] + n_2 * dims_poisson_kernel_error[0][0] * dims_poisson_kernel_error[0][1] + n_3 * dims_poisson_kernel_error[0][0] * dims_poisson_kernel_error[0][1] * dims_poisson_kernel_error[0][2]);
    const ACC<float> ref(dims_poisson_kernel_error[1][0], dims_poisson_kernel_error[1][1], dims_poisson_kernel_error[1][2], ref_p + n_0 + n_1 * dims_poisson_kernel_error[1][0] + n_2 * dims_poisson_kernel_error[1][0] * dims_poisson_kernel_error[1][1] + n_3 * dims_poisson_kernel_error[1][0] * dims_poisson_kernel_error[1][1] * dims_poisson_kernel_error[1][2]);
    
  *err = *err + (u(0,0,0)-ref(0,0,0))*(u(0,0,0)-ref(0,0,0));

  }
  for (int d=0; d<1; d++)
  #if defined(OPS_BATCHED) && OPS_BATCHED==0
    ops_reduction_cuda<OPS_INC>(&err_p[d+n_0*1],err[d],blockDim.y*blockDim.z*blockDim.u, blockDim.x, 0);
  #elif defined(OPS_BATCHED) && OPS_BATCHED==3
    ops_reduction_cuda<OPS_INC>(&err_p[d+n_3*1],err[d],blockDim.x*blockDim.y*blockDim.z, 1, 0);
  #elif defined(OPS_BATCHED)
  #error CUDA Reductions only implemented for OPS_BATCHED=0 or 3
  #else
    ops_reduction_cuda<OPS_INC>(&err_p[d+(blockIdx.x + blockIdx.y*gridDim.x + blockIdx.z*gridDim.x*gridDim.y)*1],err[d]);
  #endif

}

// host stub function
#ifndef OPS_LAZY
void ops_par_loop_poisson_kernel_error(char const *name, ops_block block, int dim, int* range,
 ops_arg arg0, ops_arg arg1, ops_arg arg2) {
const int blockidx_start = 0; const int blockidx_end = block->count;
#ifdef OPS_BATCHED
const int batch_size = block->count;
#endif
#else
void ops_par_loop_poisson_kernel_error_execute(const char *name, ops_block block, int blockidx_start, int blockidx_end, int dim, int *range, int nargs, ops_arg* args) {
  #ifdef OPS_BATCHED
  const int batch_size = OPS_BATCH_SIZE;
  #endif
  ops_arg arg0 = args[0];
  ops_arg arg1 = args[1];
  ops_arg arg2 = args[2];
  #endif

  //Timing
  double __t1,__t2,__c1,__c2;

  #ifndef OPS_LAZY
  ops_arg args[3] = { arg0, arg1, arg2};


  #endif

  #if defined(CHECKPOINTING) && !defined(OPS_LAZY)
  if (!ops_checkpointing_before(args,3,range,5)) return;
  #endif

  if (OPS_instance::getOPSInstance()->OPS_diags > 1) {
    ops_timing_realloc(5,"poisson_kernel_error");
    OPS_instance::getOPSInstance()->OPS_kernels[5].count++;
    ops_timers_core(&__c2,&__t2);
  }

  #ifdef OPS_DEBUG
  ops_register_args(args, "poisson_kernel_error");
  #endif


  //compute locally allocated range for the sub-block
  int start[3];
  int end[3];
  #ifdef OPS_MPI
  int arg_idx[3];
  #endif
  #if defined(OPS_LAZY) || !defined(OPS_MPI)
  for ( int n=0; n<3; n++ ){
    start[n] = range[2*n];end[n] = range[2*n+1];
  }
  #else
  if (compute_ranges(args, 3,block, range, start, end, arg_idx) < 0) return;
  #endif


  #ifdef OPS_BATCHED
  const int bounds_0_l = OPS_BATCHED == 0 ? 0 : start[(OPS_BATCHED>0)+-1];
  const int bounds_0_u = OPS_BATCHED == 0 ? MIN(batch_size,block->count-blockidx_start) : end[(OPS_BATCHED>0)+-1];
  const int bounds_1_l = OPS_BATCHED == 1 ? 0 : start[(OPS_BATCHED>1)+0];
  const int bounds_1_u = OPS_BATCHED == 1 ? MIN(batch_size,block->count-blockidx_start) : end[(OPS_BATCHED>1)+0];
  const int bounds_2_l = OPS_BATCHED == 2 ? 0 : start[(OPS_BATCHED>2)+1];
  const int bounds_2_u = OPS_BATCHED == 2 ? MIN(batch_size,block->count-blockidx_start) : end[(OPS_BATCHED>2)+1];
  const int bounds_3_l = OPS_BATCHED == 3 ? 0 : start[(OPS_BATCHED>3)+2];
  const int bounds_3_u = OPS_BATCHED == 3 ? MIN(batch_size,block->count-blockidx_start) : end[(OPS_BATCHED>3)+2];
  #else
  const int bounds_0_l = start[0];
  const int bounds_0_u = end[0];
  const int bounds_1_l = start[1];
  const int bounds_1_u = end[1];
  const int bounds_2_l = start[2];
  const int bounds_2_u = end[2];
  const int bounds_3_l = 0;
  const int bounds_3_u = blockidx_end-blockidx_start;
  #endif
  if (args[0].dat->size[0] != dims_poisson_kernel_error_h[0][0] || args[0].dat->size[1] != dims_poisson_kernel_error_h[0][1] || args[0].dat->size[2] != dims_poisson_kernel_error_h[0][2] || args[0].dat->size[3] != dims_poisson_kernel_error_h[0][3] || args[1].dat->size[0] != dims_poisson_kernel_error_h[1][0] || args[1].dat->size[1] != dims_poisson_kernel_error_h[1][1] || args[1].dat->size[2] != dims_poisson_kernel_error_h[1][2] || args[1].dat->size[3] != dims_poisson_kernel_error_h[1][3]) {
    dims_poisson_kernel_error_h[0][0] = args[0].dat->size[0];
    dims_poisson_kernel_error_h[0][1] = args[0].dat->size[1];
    dims_poisson_kernel_error_h[0][2] = args[0].dat->size[2];
    dims_poisson_kernel_error_h[0][3] = args[0].dat->size[3];
    dims_poisson_kernel_error_h[1][0] = args[1].dat->size[0];
    dims_poisson_kernel_error_h[1][1] = args[1].dat->size[1];
    dims_poisson_kernel_error_h[1][2] = args[1].dat->size[2];
    dims_poisson_kernel_error_h[1][3] = args[1].dat->size[3];
    cutilSafeCall(cudaMemcpyToSymbol( dims_poisson_kernel_error, dims_poisson_kernel_error_h, sizeof(dims_poisson_kernel_error)));
  }

  //set up initial pointers
  float * __restrict__ u_p = (float *)(args[0].data_d + args[0].dat->base_offset + blockidx_start * args[0].dat->batch_offset);

  float * __restrict__ ref_p = (float *)(args[1].data_d + args[1].dat->base_offset + blockidx_start * args[1].dat->batch_offset);

  #ifdef OPS_MPI
  float * __restrict__ p_a2 = (float *)(((ops_reduction)args[2].data)->data + ((ops_reduction)args[2].data)->size * block->index + ((ops_reduction)args[2].data)->size * blockidx_start);
  #else //OPS_MPI
  float * __restrict__ p_a2 = (float *)(((ops_reduction)args[2].data)->data + ((ops_reduction)args[2].data)->size * blockidx_start);
  #endif //OPS_MPI




  int x_size = MAX(0,bounds_0_u-bounds_0_l);
  int y_size = MAX(0,bounds_1_u-bounds_1_l);
  int z_size = MAX(0,bounds_2_u-bounds_2_l);
  z_size *= MAX(0,bounds_3_u-bounds_3_l);

  dim3 grid( (x_size-1)/OPS_instance::getOPSInstance()->OPS_block_size_x+ 1, (y_size-1)/OPS_instance::getOPSInstance()->OPS_block_size_y + 1, (z_size-1)/OPS_instance::getOPSInstance()->OPS_block_size_z+1);
  dim3 tblock(MIN(OPS_instance::getOPSInstance()->OPS_block_size_x, x_size), MIN(OPS_instance::getOPSInstance()->OPS_block_size_y, y_size),MIN(OPS_instance::getOPSInstance()->OPS_block_size_z, z_size));

  #ifdef OPS_BATCHED
  int nblocks = blockidx_end-blockidx_start;
  #else
  int nblocks = grid.x * grid.y * grid.z;
  #endif
  int reduct_bytes = 0;
  int reduct_size = 0;

  reduct_bytes += ROUND_UP(nblocks*1*sizeof(float));
  reduct_size = MAX(reduct_size,sizeof(float)*1);

  reallocReductArrays(reduct_bytes);
  reduct_bytes = 0;

  arg2.data = OPS_instance::getOPSInstance()->OPS_reduct_h + reduct_bytes;
  arg2.data_d = OPS_instance::getOPSInstance()->OPS_reduct_d + reduct_bytes;
  for (int b=0; b<nblocks; b++)
  for (int d=0; d<1; d++) ((float *)arg2.data)[d+b*1] = ZERO_float;
  reduct_bytes += ROUND_UP(nblocks*1*sizeof(float));


  mvReductArraysToDevice(reduct_bytes);
  #ifndef OPS_LAZY
  //Halo Exchanges
  ops_H_D_exchanges_device(args, 3);
  ops_halo_exchanges(args,3,range);
  ops_H_D_exchanges_device(args, 3);
  #endif

  if (OPS_instance::getOPSInstance()->OPS_diags > 1) {
    ops_timers_core(&__c1,&__t1);
    OPS_instance::getOPSInstance()->OPS_kernels[5].mpi_time += __t1-__t2;
  }

  int nshared = 0;
  int nthread = OPS_instance::getOPSInstance()->OPS_block_size_x*OPS_instance::getOPSInstance()->OPS_block_size_y*OPS_instance::getOPSInstance()->OPS_block_size_z;

  nshared = MAX(nshared,sizeof(float)*1);

  nshared = MAX(nshared*nthread,reduct_size*nthread);

  //call kernel wrapper function, passing in pointers to data
  if (x_size > 0 && y_size > 0 && z_size > 0)
    ops_poisson_kernel_error<<<grid, tblock, nshared >>> (  u_p, ref_p,
         (float *)arg2.data_d,         bounds_0_l, bounds_0_u, bounds_1_l, bounds_1_u,
         bounds_2_l, bounds_2_u, bounds_3_l, bounds_3_u);

  cutilSafeCall(cudaGetLastError());

  mvReductArraysToHost(reduct_bytes);
  for ( int b=0; b<nblocks; b++ ){
    for ( int d=0; d<1; d++ ){
      #ifdef OPS_BATCHED
      p_a2[b*1 + d] = p_a2[b*1 + d] + ((float *)arg2.data)[d+b*1];
      #else
      p_a2[d] = p_a2[d] + ((float *)arg2.data)[d+b*1];
      #endif
    }
  }
  arg2.data = (char *)p_a2;

  if (OPS_instance::getOPSInstance()->OPS_diags>1) {
    cutilSafeCall(cudaDeviceSynchronize());
  }

  if (OPS_instance::getOPSInstance()->OPS_diags > 1) {
    ops_timers_core(&__c2,&__t2);
    OPS_instance::getOPSInstance()->OPS_kernels[5].time += __t2-__t1;
  }
  #ifndef OPS_LAZY
  ops_set_dirtybit_device(args, 3);
  #endif

  if (OPS_instance::getOPSInstance()->OPS_diags > 1) {
    //Update kernel record
    ops_timers_core(&__c1,&__t1);
    OPS_instance::getOPSInstance()->OPS_kernels[5].mpi_time += __t1-__t2;
    OPS_instance::getOPSInstance()->OPS_kernels[5].transfer += ops_compute_transfer(dim, start, end, &arg0);
    OPS_instance::getOPSInstance()->OPS_kernels[5].transfer += ops_compute_transfer(dim, start, end, &arg1);
  }
}

#ifdef OPS_LAZY
void ops_par_loop_poisson_kernel_error(char const *name, ops_block block, int dim, int* range,
 ops_arg arg0, ops_arg arg1, ops_arg arg2) {
  ops_kernel_descriptor *desc = (ops_kernel_descriptor *)malloc(sizeof(ops_kernel_descriptor));
  desc->name = name;
  desc->block = block;
  desc->dim = dim;
  desc->device = 1;
  desc->index = 5;
  desc->hash = 5381;
  desc->hash = ((desc->hash << 5) + desc->hash) + 5;
  for ( int i=0; i<6; i++ ){
    desc->range[i] = range[i];
    desc->orig_range[i] = range[i];
    desc->hash = ((desc->hash << 5) + desc->hash) + range[i];
  }
  desc->nargs = 3;
  desc->args = (ops_arg*)malloc(3*sizeof(ops_arg));
  desc->args[0] = arg0;
  desc->hash = ((desc->hash << 5) + desc->hash) + arg0.dat->index;
  desc->args[1] = arg1;
  desc->hash = ((desc->hash << 5) + desc->hash) + arg1.dat->index;
  desc->args[2] = arg2;
  desc->function = ops_par_loop_poisson_kernel_error_execute;
  if (OPS_instance::getOPSInstance()->OPS_diags > 1) {
    ops_timing_realloc(5,"poisson_kernel_error");
  }
  ops_enqueue_kernel(desc);
}
#endif
