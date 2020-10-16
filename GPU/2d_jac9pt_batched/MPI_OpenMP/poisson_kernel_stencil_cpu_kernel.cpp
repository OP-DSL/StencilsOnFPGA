//
// auto-generated by ops.py
//


// host stub function
#ifndef OPS_LAZY
void ops_par_loop_poisson_kernel_stencil(char const *name, ops_block block, int dim, int* range,
 ops_arg arg0, ops_arg arg1) {
const int blockidx_start = 0; const int blockidx_end = block->count;
#ifdef OPS_BATCHED
const int batch_size = block->count;
#endif
#else
void ops_par_loop_poisson_kernel_stencil_execute(const char *name, ops_block block, int blockidx_start, int blockidx_end, int dim, int *range, int nargs, ops_arg* args) {
  #ifdef OPS_BATCHED
  const int batch_size = OPS_BATCH_SIZE;
  #endif
  ops_arg arg0 = args[0];
  ops_arg arg1 = args[1];
  #endif

  //Timing
  double __t1,__t2,__c1,__c2;

  #ifndef OPS_LAZY
  ops_arg args[2] = { arg0, arg1};


  #endif

  #if defined(CHECKPOINTING) && !defined(OPS_LAZY)
  if (!ops_checkpointing_before(args,2,range,3)) return;
  #endif

  if (OPS_instance::getOPSInstance()->OPS_diags > 1) {
    ops_timing_realloc(3,"poisson_kernel_stencil");
    OPS_instance::getOPSInstance()->OPS_kernels[3].count++;
    ops_timers_core(&__c2,&__t2);
  }

  #ifdef OPS_DEBUG
  ops_register_args(args, "poisson_kernel_stencil");
  #endif


  //compute locally allocated range for the sub-block
  int start[2];
  int end[2];
  #ifdef OPS_MPI
  int arg_idx[2];
  #endif
  #if defined(OPS_LAZY) || !defined(OPS_MPI)
  for ( int n=0; n<2; n++ ){
    start[n] = range[2*n];end[n] = range[2*n+1];
  }
  #else
  if (compute_ranges(args, 2,block, range, start, end, arg_idx) < 0) return;
  #endif


  //initialize variable with the dimension of dats
  #if defined(OPS_BATCHED) && OPS_BATCHED==0 && defined(OPS_HYBRID_LAYOUT)
  const int xdim0 = OPS_BATCH_SIZE;
  const int xdim1 = OPS_BATCH_SIZE;
  #else
  const int xdim0 = args[0].dat->size[0];
  const int xdim1 = args[1].dat->size[0];
  #endif
  const int ydim0 = args[0].dat->size[1];
  const int ydim1 = args[1].dat->size[1];
  #ifdef OPS_BATCHED
  const int bounds_0_l = OPS_BATCHED == 0 ? 0 : start[(OPS_BATCHED>0)+-1];
  const int bounds_0_u = OPS_BATCHED == 0 ? MIN(batch_size,block->count-blockidx_start) : end[(OPS_BATCHED>0)+-1];
  const int bounds_1_l = OPS_BATCHED == 1 ? 0 : start[(OPS_BATCHED>1)+0];
  const int bounds_1_u = OPS_BATCHED == 1 ? MIN(batch_size,block->count-blockidx_start) : end[(OPS_BATCHED>1)+0];
  const int bounds_2_l = OPS_BATCHED == 2 ? 0 : start[(OPS_BATCHED>2)+1];
  const int bounds_2_u = OPS_BATCHED == 2 ? MIN(batch_size,block->count-blockidx_start) : end[(OPS_BATCHED>2)+1];
  #else
  const int bounds_0_l = start[0];
  const int bounds_0_u = end[0];
  const int bounds_1_l = start[1];
  const int bounds_1_u = end[1];
  const int bounds_2_l = 0;
  const int bounds_2_u = blockidx_end-blockidx_start;
  #endif

  #ifndef OPS_LAZY
  //Halo Exchanges
  ops_H_D_exchanges_host(args, 2);
  ops_halo_exchanges(args,2,range);
  ops_H_D_exchanges_host(args, 2);
  #endif

  if (OPS_instance::getOPSInstance()->OPS_diags > 1) {
    ops_timers_core(&__c1,&__t1);
    OPS_instance::getOPSInstance()->OPS_kernels[3].mpi_time += __t1-__t2;
  }

  //set up initial pointers
  float * __restrict__ u_p = (float *)(args[0].data + args[0].dat->base_offset + blockidx_start * args[0].dat->batch_offset);

  float * __restrict__ u2_p = (float *)(args[1].data + args[1].dat->base_offset + blockidx_start * args[1].dat->batch_offset);


  #if defined(_OPENMP) && defined(OPS_BATCHED) && !defined(OPS_LAZY)
  #pragma omp parallel for
  #endif
  for ( int n_2=bounds_2_l; n_2<bounds_2_u; n_2++ ){
    #if defined(_OPENMP) && !defined(OPS_BATCHED)
    #pragma omp parallel for
    #endif
    for ( int n_1=bounds_1_l; n_1<bounds_1_u; n_1++ ){
      #ifdef __INTEL_COMPILER
      #pragma loop_count(10000)
      #pragma omp simd
      #elif defined(__clang__)
      #pragma clang loop vectorize(assume_safety)
      #elif defined(__GNUC__)
      #pragma simd
      #pragma GCC ivdep
      #else
      #pragma simd
      #endif
      for ( int n_0=bounds_0_l; n_0<bounds_0_u; n_0++ ){
        const ACC<float> u(xdim0, ydim0, u_p + n_0 + n_1 * xdim0 + n_2 * xdim0 * ydim0);
        ACC<float> u2(xdim1, ydim1, u2_p + n_0 + n_1 * xdim1 + n_2 * xdim1 * ydim1);
        
  u2(0,0) = u(-1,1)*(-0.07f) + u(0,1) * (-0.06f) + u(1,1)*(-0.05f) \
	    + u(-1,0)*(-0.08f) + u(0,0)*0.36f + u(1,0)*(-0.04f) + \
	    u(-1,-1)*(-0.01f) + u(0,-1)*(-0.02f) + u(1,-1)*(-0.03f);

      }
    }
    #if OPS_BATCHED==2 || !defined(OPS_BATCHED)
    #endif
  }
  if (OPS_instance::getOPSInstance()->OPS_diags > 1) {
    ops_timers_core(&__c2,&__t2);
    OPS_instance::getOPSInstance()->OPS_kernels[3].time += __t2-__t1;
  }
  #ifndef OPS_LAZY
  ops_set_dirtybit_host(args, 2);
  ops_set_halo_dirtybit3(&args[1],range);
  #endif

  if (OPS_instance::getOPSInstance()->OPS_diags > 1) {
    //Update kernel record
    ops_timers_core(&__c1,&__t1);
    OPS_instance::getOPSInstance()->OPS_kernels[3].mpi_time += __t1-__t2;
    OPS_instance::getOPSInstance()->OPS_kernels[3].transfer += ops_compute_transfer(dim, start, end, &arg0);
    OPS_instance::getOPSInstance()->OPS_kernels[3].transfer += ops_compute_transfer(dim, start, end, &arg1);
  }
}


#ifdef OPS_LAZY
void ops_par_loop_poisson_kernel_stencil(char const *name, ops_block block, int dim, int* range,
 ops_arg arg0, ops_arg arg1) {
  ops_kernel_descriptor *desc = (ops_kernel_descriptor *)malloc(sizeof(ops_kernel_descriptor));
  desc->name = name;
  desc->block = block;
  desc->dim = dim;
  desc->device = 1;
  desc->index = 3;
  desc->hash = 5381;
  desc->hash = ((desc->hash << 5) + desc->hash) + 3;
  for ( int i=0; i<4; i++ ){
    desc->range[i] = range[i];
    desc->orig_range[i] = range[i];
    desc->hash = ((desc->hash << 5) + desc->hash) + range[i];
  }
  desc->nargs = 2;
  desc->args = (ops_arg*)malloc(2*sizeof(ops_arg));
  desc->args[0] = arg0;
  desc->hash = ((desc->hash << 5) + desc->hash) + arg0.dat->index;
  desc->args[1] = arg1;
  desc->hash = ((desc->hash << 5) + desc->hash) + arg1.dat->index;
  desc->function = ops_par_loop_poisson_kernel_stencil_execute;
  if (OPS_instance::getOPSInstance()->OPS_diags > 1) {
    ops_timing_realloc(3,"poisson_kernel_stencil");
  }
  ops_enqueue_kernel(desc);
}
#endif
