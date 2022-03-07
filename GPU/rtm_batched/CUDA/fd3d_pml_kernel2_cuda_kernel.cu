//
// auto-generated by ops.py
//
__constant__ int dims_fd3d_pml_kernel2[13][4];
static int dims_fd3d_pml_kernel2_h[13][4] = {0};

// user function

__global__ void ops_fd3d_pml_kernel2(
    const int dispx_p, const int dispy_p, const int dispz_p, const float dt_p,
    const float scale1_p, const float scale2_p, float *__restrict rho_p,
    float *__restrict mu_p, float *__restrict yy_p, float *__restrict dyyIn_p,
    float *__restrict dyyOut_p, float *__restrict sum_p, int blockidx_start,
#ifdef OPS_MPI
    int arg_idx0, int arg_idx1, int arg_idx2,
#endif
    int bounds_0_l, int bounds_0_u, int bounds_1_l, int bounds_1_u,
    int bounds_2_l, int bounds_2_u, int bounds_3_l, int bounds_3_u) {

  const int *__restrict__ dispx = &dispx_p;
  const int *__restrict__ dispy = &dispy_p;
  const int *__restrict__ dispz = &dispz_p;
  const float *__restrict__ dt = &dt_p;
  const float *__restrict__ scale1 = &scale1_p;
  const float *__restrict__ scale2 = &scale2_p;

  int n_2 = bounds_2_l + blockDim.z * blockIdx.z + threadIdx.z;
  int n_3 = n_2 / (bounds_2_u - bounds_2_l);
#ifdef OPS_BATCHED
  n_2 = n_2 % (bounds_2_u - bounds_2_l);
#endif
                  int n_1 = bounds_1_l + blockDim.y * blockIdx.y + threadIdx.y;
  int n_0 = bounds_0_l + blockDim.x * blockIdx.x + threadIdx.x;

  int arg_idx[4] = {0};
#ifdef OPS_MPI
  arg_idx[0] = arg_idx0;
  arg_idx[1] = arg_idx1;
  arg_idx[2] = arg_idx2;
#endif
#if defined(OPS_BATCHED) && OPS_BATCHED == 0
  int idx[] = {arg_idx[0] + n_1, arg_idx[1] + n_2, arg_idx[2] + n_3,
               blockidx_start + n_0};
#elif OPS_BATCHED == 1
  int idx[] = {arg_idx[0] + n_0, arg_idx[1] + n_2, arg_idx[2] + n_3,
               blockidx_start + n_1};
#elif OPS_BATCHED == 2
  int idx[] = {arg_idx[0] + n_0, arg_idx[1] + n_1, arg_idx[2] + n_3,
               blockidx_start + n_2};
#else
  int idx[] = {arg_idx[0] + n_0, arg_idx[1] + n_1, arg_idx[2] + n_2,
               blockidx_start + n_3};
#endif
  if (n_0 < bounds_0_u && n_1 < bounds_1_u && n_2 < bounds_2_u &&
      n_3 < bounds_3_u) {
    const ACC<float> rho(
        dims_fd3d_pml_kernel2[7][0], dims_fd3d_pml_kernel2[7][1],
        dims_fd3d_pml_kernel2[7][2],
        rho_p + n_0 + n_1 * dims_fd3d_pml_kernel2[7][0] +
            n_2 * dims_fd3d_pml_kernel2[7][0] * dims_fd3d_pml_kernel2[7][1] +
            n_3 * dims_fd3d_pml_kernel2[7][0] * dims_fd3d_pml_kernel2[7][1] *
                dims_fd3d_pml_kernel2[7][2]);
    const ACC<float> mu(
        dims_fd3d_pml_kernel2[8][0], dims_fd3d_pml_kernel2[8][1],
        dims_fd3d_pml_kernel2[8][2],
        mu_p + n_0 + n_1 * dims_fd3d_pml_kernel2[8][0] +
            n_2 * dims_fd3d_pml_kernel2[8][0] * dims_fd3d_pml_kernel2[8][1] +
            n_3 * dims_fd3d_pml_kernel2[8][0] * dims_fd3d_pml_kernel2[8][1] *
                dims_fd3d_pml_kernel2[8][2]);
#ifdef OPS_SOA
    const ACC<float> yy(
        6, dims_fd3d_pml_kernel2[9][0], dims_fd3d_pml_kernel2[9][1],
        dims_fd3d_pml_kernel2[9][2],
        yy_p + n_0 + n_1 * dims_fd3d_pml_kernel2[9][0] +
            n_2 * dims_fd3d_pml_kernel2[9][0] * dims_fd3d_pml_kernel2[9][1] +
            n_3 * dims_fd3d_pml_kernel2[9][0] * dims_fd3d_pml_kernel2[9][1] *
                dims_fd3d_pml_kernel2[9][2]);
#else
    const ACC<float> yy(6, dims_fd3d_pml_kernel2[9][0],
                        dims_fd3d_pml_kernel2[9][1],
                        dims_fd3d_pml_kernel2[9][2],
                        yy_p + 6 * (n_0 + n_1 * dims_fd3d_pml_kernel2[9][0] +
                                    n_2 * dims_fd3d_pml_kernel2[9][0] *
                                        dims_fd3d_pml_kernel2[9][1] +
                                    n_3 * dims_fd3d_pml_kernel2[9][0] *
                                        dims_fd3d_pml_kernel2[9][1] *
                                        dims_fd3d_pml_kernel2[9][2]));
#endif
#ifdef OPS_SOA
    const ACC<float> dyyIn(
        6, dims_fd3d_pml_kernel2[10][0], dims_fd3d_pml_kernel2[10][1],
        dims_fd3d_pml_kernel2[10][2],
        dyyIn_p + n_0 + n_1 * dims_fd3d_pml_kernel2[10][0] +
            n_2 * dims_fd3d_pml_kernel2[10][0] * dims_fd3d_pml_kernel2[10][1] +
            n_3 * dims_fd3d_pml_kernel2[10][0] * dims_fd3d_pml_kernel2[10][1] *
                dims_fd3d_pml_kernel2[10][2]);
#else
    const ACC<float> dyyIn(
        6, dims_fd3d_pml_kernel2[10][0], dims_fd3d_pml_kernel2[10][1],
        dims_fd3d_pml_kernel2[10][2],
        dyyIn_p + 6 * (n_0 + n_1 * dims_fd3d_pml_kernel2[10][0] +
                       n_2 * dims_fd3d_pml_kernel2[10][0] *
                           dims_fd3d_pml_kernel2[10][1] +
                       n_3 * dims_fd3d_pml_kernel2[10][0] *
                           dims_fd3d_pml_kernel2[10][1] *
                           dims_fd3d_pml_kernel2[10][2]));
#endif
#ifdef OPS_SOA
    ACC<float> dyyOut(
        6, dims_fd3d_pml_kernel2[11][0], dims_fd3d_pml_kernel2[11][1],
        dims_fd3d_pml_kernel2[11][2],
        dyyOut_p + n_0 + n_1 * dims_fd3d_pml_kernel2[11][0] +
            n_2 * dims_fd3d_pml_kernel2[11][0] * dims_fd3d_pml_kernel2[11][1] +
            n_3 * dims_fd3d_pml_kernel2[11][0] * dims_fd3d_pml_kernel2[11][1] *
                dims_fd3d_pml_kernel2[11][2]);
#else
    ACC<float> dyyOut(6, dims_fd3d_pml_kernel2[11][0],
                      dims_fd3d_pml_kernel2[11][1],
                      dims_fd3d_pml_kernel2[11][2],
                      dyyOut_p + 6 * (n_0 + n_1 * dims_fd3d_pml_kernel2[11][0] +
                                      n_2 * dims_fd3d_pml_kernel2[11][0] *
                                          dims_fd3d_pml_kernel2[11][1] +
                                      n_3 * dims_fd3d_pml_kernel2[11][0] *
                                          dims_fd3d_pml_kernel2[11][1] *
                                          dims_fd3d_pml_kernel2[11][2]));
#endif
#ifdef OPS_SOA
    ACC<float> sum(
        6, dims_fd3d_pml_kernel2[12][0], dims_fd3d_pml_kernel2[12][1],
        dims_fd3d_pml_kernel2[12][2],
        sum_p + n_0 + n_1 * dims_fd3d_pml_kernel2[12][0] +
            n_2 * dims_fd3d_pml_kernel2[12][0] * dims_fd3d_pml_kernel2[12][1] +
            n_3 * dims_fd3d_pml_kernel2[12][0] * dims_fd3d_pml_kernel2[12][1] *
                dims_fd3d_pml_kernel2[12][2]);
#else
    ACC<float> sum(6, dims_fd3d_pml_kernel2[12][0],
                   dims_fd3d_pml_kernel2[12][1], dims_fd3d_pml_kernel2[12][2],
                   sum_p + 6 * (n_0 + n_1 * dims_fd3d_pml_kernel2[12][0] +
                                n_2 * dims_fd3d_pml_kernel2[12][0] *
                                    dims_fd3d_pml_kernel2[12][1] +
                                n_3 * dims_fd3d_pml_kernel2[12][0] *
                                    dims_fd3d_pml_kernel2[12][1] *
                                    dims_fd3d_pml_kernel2[12][2]));
#endif

    const float c[9] = {
        0.0035714285714285713, -0.0380952380952381,   0.2, -0.8, 0.0, 0.8, -0.2,
        0.0380952380952381,    -0.0035714285714285713};
    float invdx = 1.0 / dx;
    float invdy = 1.0 / dy;
    float invdz = 1.0 / dz;
    int xbeg = half;
    int xend = nx - half;
    int ybeg = half;
    int yend = ny - half;
    int zbeg = half;
    int zend = nz - half;
    int xpmlbeg = xbeg + pml_width;
    int ypmlbeg = ybeg + pml_width;
    int zpmlbeg = zbeg + pml_width;
    int xpmlend = xend - pml_width;
    int ypmlend = yend - pml_width;
    int zpmlend = zend - pml_width;

    float sigma = mu(0, 0, 0) / rho(0, 0, 0);
    float sigmax = 0.0;
    float sigmay = 0.0;
    float sigmaz = 0.0;
    if (idx[0] <= xbeg + pml_width) {
      sigmax = (xbeg + pml_width - idx[0]) * sigma * 0.1f;
    }
    if (idx[0] >= xend - pml_width) {
      sigmax = (idx[0] - (xend - pml_width)) * sigma * 0.1f;
    }
    if (idx[1] <= ybeg + pml_width) {
      sigmay = (ybeg + pml_width - idx[1]) * sigma * 0.1f;
    }
    if (idx[1] >= yend - pml_width) {
      sigmay = (idx[1] - (yend - pml_width)) * sigma * 0.1f;
    }
    if (idx[2] <= zbeg + pml_width) {
      sigmaz = (zbeg + pml_width - idx[2]) * sigma * 0.1f;
    }
    if (idx[2] >= zend - pml_width) {
      sigmaz = (idx[2] - (zend - pml_width)) * sigma * 0.1f;
    }

    float px = dyyIn(0, 0, 0, 0);
    float py = dyyIn(1, 0, 0, 0);
    float pz = dyyIn(2, 0, 0, 0);

    float vx = dyyIn(3, 0, 0, 0);
    float vy = dyyIn(4, 0, 0, 0);
    float vz = dyyIn(5, 0, 0, 0);

    float vxx = 0.0;
    float vxy = 0.0;
    float vxz = 0.0;

    float vyx = 0.0;
    float vyy = 0.0;
    float vyz = 0.0;

    float vzx = 0.0;
    float vzy = 0.0;
    float vzz = 0.0;

    float pxx = 0.0;
    float pxy = 0.0;
    float pxz = 0.0;

    float pyx = 0.0;
    float pyy = 0.0;
    float pyz = 0.0;

    float pzx = 0.0;
    float pzy = 0.0;
    float pzz = 0.0;

    for (int i = -half; i <= half; i++) {
      pxx += dyyIn(0, i, 0, 0) * c[i + half];
      pyx += dyyIn(1, i, 0, 0) * c[i + half];
      pzx += dyyIn(2, i, 0, 0) * c[i + half];

      vxx += dyyIn(3, i, 0, 0) * c[i + half];
      vyx += dyyIn(4, i, 0, 0) * c[i + half];
      vzx += dyyIn(5, i, 0, 0) * c[i + half];

      pxy += dyyIn(0, 0, i, 0) * c[i + half];
      pyy += dyyIn(1, 0, i, 0) * c[i + half];
      pzy += dyyIn(2, 0, i, 0) * c[i + half];

      vxy += dyyIn(3, 0, i, 0) * c[i + half];
      vyy += dyyIn(4, 0, i, 0) * c[i + half];
      vzy += dyyIn(5, 0, i, 0) * c[i + half];

      pxz += dyyIn(0, 0, 0, i) * c[i + half];
      pyz += dyyIn(1, 0, 0, i) * c[i + half];
      pzz += dyyIn(2, 0, 0, i) * c[i + half];

      vxz += dyyIn(3, 0, 0, i) * c[i + half];
      vyz += dyyIn(4, 0, 0, i) * c[i + half];
      vzz += dyyIn(5, 0, 0, i) * c[i + half];
    }

    pxx *= invdx;
    pyx *= invdx;
    pzx *= invdx;

    vxx *= invdx;
    vyx *= invdx;
    vzx *= invdx;

    pxy *= invdy;
    pyy *= invdy;
    pzy *= invdy;

    vxy *= invdy;
    vyy *= invdy;
    vzy *= invdy;

    pxz *= invdz;
    pyz *= invdz;
    pzz *= invdz;

    vxz *= invdz;
    vyz *= invdz;
    vzz *= invdz;

    float ytemp0 = (vxx / rho(0, 0, 0) - sigmax * px) * *dt;
    float ytemp3 = ((pxx + pyx + pxz) * mu(0, 0, 0) - sigmax * vx) * *dt;

    float ytemp1 = (vyy / rho(0, 0, 0) - sigmay * py) * *dt;
    float ytemp4 = ((pxy + pyy + pyz) * mu(0, 0, 0) - sigmay * vy) * *dt;

    float ytemp2 = (vzz / rho(0, 0, 0) - sigmaz * pz) * *dt;
    float ytemp5 = ((pxz + pyz + pzz) * mu(0, 0, 0) - sigmaz * vz) * *dt;

    dyyOut(0, 0, 0, 0) = yy(0, 0, 0, 0) + ytemp0 * *scale1;
    dyyOut(3, 0, 0, 0) = yy(3, 0, 0, 0) + ytemp3 * *scale1;
    dyyOut(1, 0, 0, 0) = yy(1, 0, 0, 0) + ytemp1 * *scale1;
    dyyOut(4, 0, 0, 0) = yy(4, 0, 0, 0) + ytemp4 * *scale1;
    dyyOut(2, 0, 0, 0) = yy(2, 0, 0, 0) + ytemp2 * *scale1;
    dyyOut(5, 0, 0, 0) = yy(5, 0, 0, 0) + ytemp5 * *scale1;

    sum(0, 0, 0, 0) += ytemp0 * *scale2;
    sum(3, 0, 0, 0) += ytemp3 * *scale2;
    sum(1, 0, 0, 0) += ytemp1 * *scale2;
    sum(4, 0, 0, 0) += ytemp4 * *scale2;
    sum(2, 0, 0, 0) += ytemp2 * *scale2;
    sum(5, 0, 0, 0) += ytemp5 * *scale2;
  }
}

// host stub function
#ifndef OPS_LAZY
void ops_par_loop_fd3d_pml_kernel2(char const *name, ops_block block, int dim,
                                   int *range, ops_arg arg0, ops_arg arg1,
                                   ops_arg arg2, ops_arg arg3, ops_arg arg4,
                                   ops_arg arg5, ops_arg arg6, ops_arg arg7,
                                   ops_arg arg8, ops_arg arg9, ops_arg arg10,
                                   ops_arg arg11, ops_arg arg12) {
  const int blockidx_start = 0;
  const int blockidx_end = block->count;
#ifdef OPS_BATCHED
  const int batch_size = block->count;
#endif
#else
void ops_par_loop_fd3d_pml_kernel2_execute(const char *name, ops_block block,
                                           int blockidx_start, int blockidx_end,
                                           int dim, int *range, int nargs,
                                           ops_arg *args) {
#ifdef OPS_BATCHED
  const int batch_size = OPS_BATCH_SIZE;
#endif
  ops_arg arg0 = args[0];
  ops_arg arg1 = args[1];
  ops_arg arg2 = args[2];
  ops_arg arg3 = args[3];
  ops_arg arg4 = args[4];
  ops_arg arg5 = args[5];
  ops_arg arg6 = args[6];
  ops_arg arg7 = args[7];
  ops_arg arg8 = args[8];
  ops_arg arg9 = args[9];
  ops_arg arg10 = args[10];
  ops_arg arg11 = args[11];
  ops_arg arg12 = args[12];
#endif

  // Timing
  double __t1, __t2, __c1, __c2;

#ifndef OPS_LAZY
  ops_arg args[13] = {arg0, arg1, arg2, arg3,  arg4,  arg5, arg6,
                      arg7, arg8, arg9, arg10, arg11, arg12};

#endif

#if defined(CHECKPOINTING) && !defined(OPS_LAZY)
  if (!ops_checkpointing_before(args, 13, range, 2))
    return;
#endif

  if (OPS_instance::getOPSInstance()->OPS_diags > 1) {
    ops_timing_realloc(2, "fd3d_pml_kernel2");
    OPS_instance::getOPSInstance()->OPS_kernels[2].count++;
    ops_timers_core(&__c2, &__t2);
  }

#ifdef OPS_DEBUG
  ops_register_args(args, "fd3d_pml_kernel2");
#endif

  // compute locally allocated range for the sub-block
  int start[3];
  int end[3];
  int arg_idx[3];
#if defined(OPS_LAZY) || !defined(OPS_MPI)
  for (int n = 0; n < 3; n++) {
    start[n] = range[2 * n];
    end[n] = range[2 * n + 1];
  }
#else
  if (compute_ranges(args, 13, block, range, start, end, arg_idx) < 0)
    return;
#endif

#ifdef OPS_MPI
  sub_dat_list sd = OPS_sub_dat_list[args[12].dat->index];
  arg_idx[0] = MAX(0, sd->decomp_disp[0]);
  arg_idx[1] = MAX(0, sd->decomp_disp[1]);
  arg_idx[2] = MAX(0, sd->decomp_disp[2]);
#else  // OPS_MPI
  arg_idx[0] = 0;
  arg_idx[1] = 0;
  arg_idx[2] = 0;
#endif // OPS_MPI

#ifdef OPS_BATCHED
  const int bounds_0_l = OPS_BATCHED == 0 ? 0 : start[(OPS_BATCHED > 0) + -1];
  const int bounds_0_u = OPS_BATCHED == 0
                             ? MIN(batch_size, block->count - blockidx_start)
                             : end[(OPS_BATCHED > 0) + -1];
  const int bounds_1_l = OPS_BATCHED == 1 ? 0 : start[(OPS_BATCHED > 1) + 0];
  const int bounds_1_u = OPS_BATCHED == 1
                             ? MIN(batch_size, block->count - blockidx_start)
                             : end[(OPS_BATCHED > 1) + 0];
  const int bounds_2_l = OPS_BATCHED == 2 ? 0 : start[(OPS_BATCHED > 2) + 1];
  const int bounds_2_u = OPS_BATCHED == 2
                             ? MIN(batch_size, block->count - blockidx_start)
                             : end[(OPS_BATCHED > 2) + 1];
  const int bounds_3_l = OPS_BATCHED == 3 ? 0 : start[(OPS_BATCHED > 3) + 2];
  const int bounds_3_u = OPS_BATCHED == 3
                             ? MIN(batch_size, block->count - blockidx_start)
                             : end[(OPS_BATCHED > 3) + 2];
#else
  const int bounds_0_l = start[0];
  const int bounds_0_u = end[0];
  const int bounds_1_l = start[1];
  const int bounds_1_u = end[1];
  const int bounds_2_l = start[2];
  const int bounds_2_u = end[2];
  const int bounds_3_l = 0;
  const int bounds_3_u = blockidx_end - blockidx_start;
#endif
  if (args[7].dat->size[0] != dims_fd3d_pml_kernel2_h[7][0] ||
      args[7].dat->size[1] != dims_fd3d_pml_kernel2_h[7][1] ||
      args[7].dat->size[2] != dims_fd3d_pml_kernel2_h[7][2] ||
      args[7].dat->size[3] != dims_fd3d_pml_kernel2_h[7][3] ||
      args[8].dat->size[0] != dims_fd3d_pml_kernel2_h[8][0] ||
      args[8].dat->size[1] != dims_fd3d_pml_kernel2_h[8][1] ||
      args[8].dat->size[2] != dims_fd3d_pml_kernel2_h[8][2] ||
      args[8].dat->size[3] != dims_fd3d_pml_kernel2_h[8][3] ||
      args[9].dat->size[0] != dims_fd3d_pml_kernel2_h[9][0] ||
      args[9].dat->size[1] != dims_fd3d_pml_kernel2_h[9][1] ||
      args[9].dat->size[2] != dims_fd3d_pml_kernel2_h[9][2] ||
      args[9].dat->size[3] != dims_fd3d_pml_kernel2_h[9][3] ||
      args[10].dat->size[0] != dims_fd3d_pml_kernel2_h[10][0] ||
      args[10].dat->size[1] != dims_fd3d_pml_kernel2_h[10][1] ||
      args[10].dat->size[2] != dims_fd3d_pml_kernel2_h[10][2] ||
      args[10].dat->size[3] != dims_fd3d_pml_kernel2_h[10][3] ||
      args[11].dat->size[0] != dims_fd3d_pml_kernel2_h[11][0] ||
      args[11].dat->size[1] != dims_fd3d_pml_kernel2_h[11][1] ||
      args[11].dat->size[2] != dims_fd3d_pml_kernel2_h[11][2] ||
      args[11].dat->size[3] != dims_fd3d_pml_kernel2_h[11][3] ||
      args[12].dat->size[0] != dims_fd3d_pml_kernel2_h[12][0] ||
      args[12].dat->size[1] != dims_fd3d_pml_kernel2_h[12][1] ||
      args[12].dat->size[2] != dims_fd3d_pml_kernel2_h[12][2] ||
      args[12].dat->size[3] != dims_fd3d_pml_kernel2_h[12][3]) {
    dims_fd3d_pml_kernel2_h[7][0] = args[7].dat->size[0];
    dims_fd3d_pml_kernel2_h[7][1] = args[7].dat->size[1];
    dims_fd3d_pml_kernel2_h[7][2] = args[7].dat->size[2];
    dims_fd3d_pml_kernel2_h[7][3] = args[7].dat->size[3];
    dims_fd3d_pml_kernel2_h[8][0] = args[8].dat->size[0];
    dims_fd3d_pml_kernel2_h[8][1] = args[8].dat->size[1];
    dims_fd3d_pml_kernel2_h[8][2] = args[8].dat->size[2];
    dims_fd3d_pml_kernel2_h[8][3] = args[8].dat->size[3];
    dims_fd3d_pml_kernel2_h[9][0] = args[9].dat->size[0];
    dims_fd3d_pml_kernel2_h[9][1] = args[9].dat->size[1];
    dims_fd3d_pml_kernel2_h[9][2] = args[9].dat->size[2];
    dims_fd3d_pml_kernel2_h[9][3] = args[9].dat->size[3];
    dims_fd3d_pml_kernel2_h[10][0] = args[10].dat->size[0];
    dims_fd3d_pml_kernel2_h[10][1] = args[10].dat->size[1];
    dims_fd3d_pml_kernel2_h[10][2] = args[10].dat->size[2];
    dims_fd3d_pml_kernel2_h[10][3] = args[10].dat->size[3];
    dims_fd3d_pml_kernel2_h[11][0] = args[11].dat->size[0];
    dims_fd3d_pml_kernel2_h[11][1] = args[11].dat->size[1];
    dims_fd3d_pml_kernel2_h[11][2] = args[11].dat->size[2];
    dims_fd3d_pml_kernel2_h[11][3] = args[11].dat->size[3];
    dims_fd3d_pml_kernel2_h[12][0] = args[12].dat->size[0];
    dims_fd3d_pml_kernel2_h[12][1] = args[12].dat->size[1];
    dims_fd3d_pml_kernel2_h[12][2] = args[12].dat->size[2];
    dims_fd3d_pml_kernel2_h[12][3] = args[12].dat->size[3];
    cutilSafeCall(cudaMemcpyToSymbol(dims_fd3d_pml_kernel2,
                                     dims_fd3d_pml_kernel2_h,
                                     sizeof(dims_fd3d_pml_kernel2)));
  }

  // set up initial pointers
  int *__restrict__ dispx = (int *)args[0].data;

  int *__restrict__ dispy = (int *)args[1].data;

  int *__restrict__ dispz = (int *)args[2].data;

  float *__restrict__ dt = (float *)args[4].data;

  float *__restrict__ scale1 = (float *)args[5].data;

  float *__restrict__ scale2 = (float *)args[6].data;

  float *__restrict__ rho_p =
      (float *)(args[7].data_d + args[7].dat->base_offset +
                blockidx_start * args[7].dat->batch_offset);

  float *__restrict__ mu_p =
      (float *)(args[8].data_d + args[8].dat->base_offset +
                blockidx_start * args[8].dat->batch_offset);

  float *__restrict__ yy_p =
      (float *)(args[9].data_d + args[9].dat->base_offset +
                blockidx_start * args[9].dat->batch_offset);

  float *__restrict__ dyyIn_p =
      (float *)(args[10].data_d + args[10].dat->base_offset +
                blockidx_start * args[10].dat->batch_offset);

  float *__restrict__ dyyOut_p =
      (float *)(args[11].data_d + args[11].dat->base_offset +
                blockidx_start * args[11].dat->batch_offset);

  float *__restrict__ sum_p =
      (float *)(args[12].data_d + args[12].dat->base_offset +
                blockidx_start * args[12].dat->batch_offset);

  int x_size = MAX(0, bounds_0_u - bounds_0_l);
  int y_size = MAX(0, bounds_1_u - bounds_1_l);
  int z_size = MAX(0, bounds_2_u - bounds_2_l);
  z_size *= MAX(0, bounds_3_u - bounds_3_l);

  dim3 grid((x_size - 1) / OPS_instance::getOPSInstance()->OPS_block_size_x + 1,
            (y_size - 1) / OPS_instance::getOPSInstance()->OPS_block_size_y + 1,
            (z_size - 1) / OPS_instance::getOPSInstance()->OPS_block_size_z +
                1);
  dim3 tblock(MIN(OPS_instance::getOPSInstance()->OPS_block_size_x, x_size),
              MIN(OPS_instance::getOPSInstance()->OPS_block_size_y, y_size),
              MIN(OPS_instance::getOPSInstance()->OPS_block_size_z, z_size));

#ifndef OPS_LAZY
  // Halo Exchanges
  ops_H_D_exchanges_device(args, 13);
  ops_halo_exchanges(args, 13, range);
  ops_H_D_exchanges_device(args, 13);
#endif

  if (OPS_instance::getOPSInstance()->OPS_diags > 1) {
    ops_timers_core(&__c1, &__t1);
    OPS_instance::getOPSInstance()->OPS_kernels[2].mpi_time += __t1 - __t2;
  }

  // call kernel wrapper function, passing in pointers to data
  if (x_size > 0 && y_size > 0 && z_size > 0)
    ops_fd3d_pml_kernel2<<<grid, tblock>>>(
        *dispx, *dispy, *dispz, *dt, *scale1, *scale2, rho_p, mu_p, yy_p,
        dyyIn_p, dyyOut_p, sum_p, blockidx_start,
#ifdef OPS_MPI
        arg_idx[0], arg_idx[1], arg_idx[2], blockidx_start,
#endif
        bounds_0_l, bounds_0_u, bounds_1_l, bounds_1_u, bounds_2_l, bounds_2_u,
        bounds_3_l, bounds_3_u);

  cutilSafeCall(cudaGetLastError());

  if (OPS_instance::getOPSInstance()->OPS_diags > 1) {
    cutilSafeCall(cudaDeviceSynchronize());
  }

  if (OPS_instance::getOPSInstance()->OPS_diags > 1) {
    ops_timers_core(&__c2, &__t2);
    OPS_instance::getOPSInstance()->OPS_kernels[2].time += __t2 - __t1;
  }
#ifndef OPS_LAZY
  ops_set_dirtybit_device(args, 13);
  ops_set_halo_dirtybit3(&args[11], range);
  ops_set_halo_dirtybit3(&args[12], range);
#endif

  if (OPS_instance::getOPSInstance()->OPS_diags > 1) {
    // Update kernel record
    ops_timers_core(&__c1, &__t1);
    OPS_instance::getOPSInstance()->OPS_kernels[2].mpi_time += __t1 - __t2;
    OPS_instance::getOPSInstance()->OPS_kernels[2].transfer +=
        ops_compute_transfer(dim, start, end, &arg7);
    OPS_instance::getOPSInstance()->OPS_kernels[2].transfer +=
        ops_compute_transfer(dim, start, end, &arg8);
    OPS_instance::getOPSInstance()->OPS_kernels[2].transfer +=
        ops_compute_transfer(dim, start, end, &arg9);
    OPS_instance::getOPSInstance()->OPS_kernels[2].transfer +=
        ops_compute_transfer(dim, start, end, &arg10);
    OPS_instance::getOPSInstance()->OPS_kernels[2].transfer +=
        ops_compute_transfer(dim, start, end, &arg11);
    OPS_instance::getOPSInstance()->OPS_kernels[2].transfer +=
        ops_compute_transfer(dim, start, end, &arg12);
  }
}

#ifdef OPS_LAZY
void ops_par_loop_fd3d_pml_kernel2(char const *name, ops_block block, int dim,
                                   int *range, ops_arg arg0, ops_arg arg1,
                                   ops_arg arg2, ops_arg arg3, ops_arg arg4,
                                   ops_arg arg5, ops_arg arg6, ops_arg arg7,
                                   ops_arg arg8, ops_arg arg9, ops_arg arg10,
                                   ops_arg arg11, ops_arg arg12) {
  ops_kernel_descriptor *desc =
      (ops_kernel_descriptor *)malloc(sizeof(ops_kernel_descriptor));
  desc->name = name;
  desc->block = block;
  desc->dim = dim;
  desc->device = 1;
  desc->index = 2;
  desc->hash = 5381;
  desc->hash = ((desc->hash << 5) + desc->hash) + 2;
  for (int i = 0; i < 6; i++) {
    desc->range[i] = range[i];
    desc->orig_range[i] = range[i];
    desc->hash = ((desc->hash << 5) + desc->hash) + range[i];
  }
  desc->nargs = 13;
  desc->args = (ops_arg *)malloc(13 * sizeof(ops_arg));
  desc->args[0] = arg0;
  char *tmp = (char *)malloc(1 * sizeof(int));
  memcpy(tmp, arg0.data, 1 * sizeof(int));
  desc->args[0].data = tmp;
  desc->args[1] = arg1;
  tmp = (char *)malloc(1 * sizeof(int));
  memcpy(tmp, arg1.data, 1 * sizeof(int));
  desc->args[1].data = tmp;
  desc->args[2] = arg2;
  tmp = (char *)malloc(1 * sizeof(int));
  memcpy(tmp, arg2.data, 1 * sizeof(int));
  desc->args[2].data = tmp;
  desc->args[3] = arg3;
  desc->args[4] = arg4;
  tmp = (char *)malloc(1 * sizeof(float));
  memcpy(tmp, arg4.data, 1 * sizeof(float));
  desc->args[4].data = tmp;
  desc->args[5] = arg5;
  tmp = (char *)malloc(1 * sizeof(float));
  memcpy(tmp, arg5.data, 1 * sizeof(float));
  desc->args[5].data = tmp;
  desc->args[6] = arg6;
  tmp = (char *)malloc(1 * sizeof(float));
  memcpy(tmp, arg6.data, 1 * sizeof(float));
  desc->args[6].data = tmp;
  desc->args[7] = arg7;
  desc->hash = ((desc->hash << 5) + desc->hash) + arg7.dat->index;
  desc->args[8] = arg8;
  desc->hash = ((desc->hash << 5) + desc->hash) + arg8.dat->index;
  desc->args[9] = arg9;
  desc->hash = ((desc->hash << 5) + desc->hash) + arg9.dat->index;
  desc->args[10] = arg10;
  desc->hash = ((desc->hash << 5) + desc->hash) + arg10.dat->index;
  desc->args[11] = arg11;
  desc->hash = ((desc->hash << 5) + desc->hash) + arg11.dat->index;
  desc->args[12] = arg12;
  desc->hash = ((desc->hash << 5) + desc->hash) + arg12.dat->index;
  desc->function = ops_par_loop_fd3d_pml_kernel2_execute;
  if (OPS_instance::getOPSInstance()->OPS_diags > 1) {
    ops_timing_realloc(2, "fd3d_pml_kernel2");
  }
  ops_enqueue_kernel(desc);
}
#endif
