//
// auto-generated by ops.py
//

extern int xdim4_rtm_kernel_populate;
int xdim4_rtm_kernel_populate_h = -1;
extern int ydim4_rtm_kernel_populate;
int ydim4_rtm_kernel_populate_h = -1;
extern int xdim5_rtm_kernel_populate;
int xdim5_rtm_kernel_populate_h = -1;
extern int ydim5_rtm_kernel_populate;
int ydim5_rtm_kernel_populate_h = -1;
extern int xdim6_rtm_kernel_populate;
int xdim6_rtm_kernel_populate_h = -1;
extern int ydim6_rtm_kernel_populate;
int ydim6_rtm_kernel_populate_h = -1;

#ifdef __cplusplus
extern "C" {
#endif
void rtm_kernel_populate_c_wrapper(
  int *p_a0,
  int *p_a1,
  int *p_a2,
  int *p_a3,
  float *p_a4,
  float *p_a5,
  float *p_a6,
  int arg_idx0, int arg_idx1, int arg_idx2,
  int x_size, int y_size, int z_size);

#ifdef __cplusplus
}
#endif

// host stub function
void ops_par_loop_rtm_kernel_populate(char const *name, ops_block block, int dim, int* range,
 ops_arg arg0, ops_arg arg1, ops_arg arg2, ops_arg arg3, ops_arg arg4, ops_arg arg5, ops_arg arg6) {

  ops_arg args[7] = { arg0, arg1, arg2, arg3, arg4, arg5, arg6};


  #ifdef CHECKPOINTING
  if (!ops_checkpointing_before(args,7,range,0)) return;
  #endif

  if (OPS_instance::getOPSInstance()->OPS_diags > 1) {
    ops_timing_realloc(0,"rtm_kernel_populate");
    OPS_instance::getOPSInstance()->OPS_kernels[0].count++;
  }

  //compute localy allocated range for the sub-block
  int start[3];
  int end[3];
  int arg_idx[3];

  #ifdef OPS_MPI
  sub_block_list sb = OPS_sub_block_list[block->index];
  if (compute_ranges(args, 7,block, range, start, end, arg_idx) < 0) return;
  #else
  for ( int n=0; n<3; n++ ){
    start[n] = range[2*n];end[n] = range[2*n+1];
    arg_idx[n] = start[n];
  }
  #endif

  int x_size = MAX(0,end[0]-start[0]);
  int y_size = MAX(0,end[1]-start[1]);
  int z_size = MAX(0,end[2]-start[2]);

  int xdim4 = args[4].dat->size[0];
  int ydim4 = args[4].dat->size[1];
  int xdim5 = args[5].dat->size[0];
  int ydim5 = args[5].dat->size[1];
  int xdim6 = args[6].dat->size[0];
  int ydim6 = args[6].dat->size[1];

  //Timing
  double t1,t2,c1,c2;
  if (OPS_instance::getOPSInstance()->OPS_diags > 1) {
    ops_timers_core(&c2,&t2);
  }

  if (xdim4 != xdim4_rtm_kernel_populate_h || ydim4 != ydim4_rtm_kernel_populate_h || xdim5 != xdim5_rtm_kernel_populate_h || ydim5 != ydim5_rtm_kernel_populate_h || xdim6 != xdim6_rtm_kernel_populate_h || ydim6 != ydim6_rtm_kernel_populate_h) {
    xdim4_rtm_kernel_populate = xdim4;
    xdim4_rtm_kernel_populate_h = xdim4;
    ydim4_rtm_kernel_populate = ydim4;
    ydim4_rtm_kernel_populate_h = ydim4;
    xdim5_rtm_kernel_populate = xdim5;
    xdim5_rtm_kernel_populate_h = xdim5;
    ydim5_rtm_kernel_populate = ydim5;
    ydim5_rtm_kernel_populate_h = ydim5;
    xdim6_rtm_kernel_populate = xdim6;
    xdim6_rtm_kernel_populate_h = xdim6;
    ydim6_rtm_kernel_populate = ydim6;
    ydim6_rtm_kernel_populate_h = ydim6;
  }


  int dat4 = (OPS_instance::getOPSInstance()->OPS_soa ? args[4].dat->type_size : args[4].dat->elem_size);
  int dat5 = (OPS_instance::getOPSInstance()->OPS_soa ? args[5].dat->type_size : args[5].dat->elem_size);
  int dat6 = (OPS_instance::getOPSInstance()->OPS_soa ? args[6].dat->type_size : args[6].dat->elem_size);

  //set up initial pointers and exchange halos if necessary
  int *p_a0 = (int *)args[0].data;


  int *p_a1 = (int *)args[1].data;


  int *p_a2 = (int *)args[2].data;


  int *p_a3 = NULL;

  int base4 = args[4].dat->base_offset + (OPS_instance::getOPSInstance()->OPS_soa ? args[4].dat->type_size : args[4].dat->elem_size) * start[0] * args[4].stencil->stride[0];
  base4 = base4+ (OPS_instance::getOPSInstance()->OPS_soa ? args[4].dat->type_size : args[4].dat->elem_size) *
    args[4].dat->size[0] *
    start[1] * args[4].stencil->stride[1];
  base4 = base4+ (OPS_instance::getOPSInstance()->OPS_soa ? args[4].dat->type_size : args[4].dat->elem_size) *
    args[4].dat->size[0] *
    args[4].dat->size[1] *
    start[2] * args[4].stencil->stride[2];
  float *p_a4 = (float *)(args[4].data + base4);

  int base5 = args[5].dat->base_offset + (OPS_instance::getOPSInstance()->OPS_soa ? args[5].dat->type_size : args[5].dat->elem_size) * start[0] * args[5].stencil->stride[0];
  base5 = base5+ (OPS_instance::getOPSInstance()->OPS_soa ? args[5].dat->type_size : args[5].dat->elem_size) *
    args[5].dat->size[0] *
    start[1] * args[5].stencil->stride[1];
  base5 = base5+ (OPS_instance::getOPSInstance()->OPS_soa ? args[5].dat->type_size : args[5].dat->elem_size) *
    args[5].dat->size[0] *
    args[5].dat->size[1] *
    start[2] * args[5].stencil->stride[2];
  float *p_a5 = (float *)(args[5].data + base5);

  int base6 = args[6].dat->base_offset + (OPS_instance::getOPSInstance()->OPS_soa ? args[6].dat->type_size : args[6].dat->elem_size) * start[0] * args[6].stencil->stride[0];
  base6 = base6+ (OPS_instance::getOPSInstance()->OPS_soa ? args[6].dat->type_size : args[6].dat->elem_size) *
    args[6].dat->size[0] *
    start[1] * args[6].stencil->stride[1];
  base6 = base6+ (OPS_instance::getOPSInstance()->OPS_soa ? args[6].dat->type_size : args[6].dat->elem_size) *
    args[6].dat->size[0] *
    args[6].dat->size[1] *
    start[2] * args[6].stencil->stride[2];
  float *p_a6 = (float *)(args[6].data + base6);



  ops_H_D_exchanges_host(args, 7);
  ops_halo_exchanges(args,7,range);

  if (OPS_instance::getOPSInstance()->OPS_diags > 1) {
    ops_timers_core(&c1,&t1);
    OPS_instance::getOPSInstance()->OPS_kernels[0].mpi_time += t1-t2;
  }

  rtm_kernel_populate_c_wrapper(
    p_a0,
    p_a1,
    p_a2,
    p_a3,
    p_a4,
    p_a5,
    p_a6,
    arg_idx[0], arg_idx[1], arg_idx[2],
    x_size, y_size, z_size);

  if (OPS_instance::getOPSInstance()->OPS_diags > 1) {
    ops_timers_core(&c2,&t2);
    OPS_instance::getOPSInstance()->OPS_kernels[0].time += t2-t1;
  }
  ops_set_dirtybit_host(args, 7);
  ops_set_halo_dirtybit3(&args[4],range);
  ops_set_halo_dirtybit3(&args[5],range);
  ops_set_halo_dirtybit3(&args[6],range);

  //Update kernel record
  if (OPS_instance::getOPSInstance()->OPS_diags > 1) {
    OPS_instance::getOPSInstance()->OPS_kernels[0].transfer += ops_compute_transfer(dim, start, end, &arg4);
    OPS_instance::getOPSInstance()->OPS_kernels[0].transfer += ops_compute_transfer(dim, start, end, &arg5);
    OPS_instance::getOPSInstance()->OPS_kernels[0].transfer += ops_compute_transfer(dim, start, end, &arg6);
  }
}
