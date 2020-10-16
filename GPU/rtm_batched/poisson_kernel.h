#ifndef poisson_KERNEL_H
#define poisson_KERNEL_H

void poisson_kernel_populate(const int *dispx, const int *dispy, const int *dispz, const int *idx, float *u, float *f, float *ref) {
  float x = dx * (float)(idx[0]+dispx[0]);
  float y = dy * (float)(idx[1]+dispy[0]);
  float z = dz * (float)(idx[2]+dispz[0]);
  printf("x = %f y = %f z = %f \n",x,y,z);
  u[OPS_ACC4(0,0,0)] = sin(M_PI*x/0.19);
  f[OPS_ACC5(0,0,0)] = -(M_PI/0.19)*(M_PI/0.19)*sin(M_PI*0.19); /* source term */
  ref[OPS_ACC6(0,0,0)] = sin(M_PI*x/0.19);
  printf("u = %f \n",u[OPS_ACC4(0,0,0)]); 
}

void rtm_kernel_populate(const int *dispx, const int *dispy, const int *dispz, const int *idx, float *rho, float *mu, float *yy) {
  //float x = dx * (float)(idx[0]+dispx[0]);
  //float y = dy * (float)(idx[1]+dispy[0]);
  //float z = dz * (float)(idx[2]+dispz[0]);
  //printf("STARTING \n");
  float x = 1.0*((float)(idx[0]-nx/2)/nx);
  float y = 1.0*((float)(idx[1]-ny/2)/ny);
  float z = 1.0*((float)(idx[2]-nz/2)/nz);
  //printf("x,y,z = %f %f %f\n",x,y,z);
  const float C = 1.0f;
  const float r0 = 0.001f;
  rho[OPS_ACC4(0,0,0)] = 1000.0f; /* density */
  mu[OPS_ACC5(0,0,0)] = 0.001f; /* bulk modulus */
  /* pressures */
  //printf("0\n");
  //printf("index = %d %d\n",OPS_ACC_MD6(0,0,0,0),OPS_ACC_MD6(1,0,0,0));
  //printf("pressx = %f\n",(1./3.)*C*exp(-(x*x+y*y+z*z)/r0));
  // float r = (static_cast <float> (rand()) - RAND_MAX/2) / static_cast <float> (RAND_MAX);

  yy[OPS_ACC_MD6(0,0,0,0)] = (1./3.)*C*exp(-(x*x+y*y+z*z)/r0); //idx[0] + idx[1] + idx[2];//
  /*
  printf("1\n");
  yy[OPS_ACC_MD6(1,0,0,0)] = (1./3.)*C*exp(-(x*x+y*y+z*z)/r0);
  printf("2\n");
  yy[OPS_ACC_MD6(2,0,0,0)] = (1./3.)*C*exp(-(x*x+y*y+z*z)/r0);
  printf("3\n");
  */
  /* velocities */
  /*
  printf("4\n");
  yy[OPS_ACC_MD6(3,0,0,0)] = 0.0;
  printf("5\n");
  yy[OPS_ACC_MD6(4,0,0,0)] = 0.0;
  printf("6\n");
  yy[OPS_ACC_MD6(5,0,0,0)] = 0.0;
  printf("DONE\n");
  */
}

void calc_ytemp_kernel(const int *dispx, const int *dispy, const int *dispz, const int *idx, const float *dt, const float* yy, float *k, float *ytemp) {

  for (int i=0;i<6;i++) {
    k[OPS_ACC_MD6(i,0,0,0)] = k[OPS_ACC_MD6(i,0,0,0)]* *dt;
    ytemp[OPS_ACC_MD7(i,0,0,0)] = yy[OPS_ACC_MD5(i,0,0,0)] + k[OPS_ACC_MD6(i,0,0,0)]*0.5f;
  }
  
}

void calc_ytemp2_kernel(const int *dispx, const int *dispy, const int *dispz, const int *idx, const float *dt, const float* yy, float *k, float *ytemp) {

  for (int i=0;i<6;i++) {
    k[OPS_ACC_MD6(i,0,0,0)] = k[OPS_ACC_MD6(i,0,0,0)]* *dt;
    ytemp[OPS_ACC_MD7(i,0,0,0)] = yy[OPS_ACC_MD5(i,0,0,0)] + k[OPS_ACC_MD6(i,0,0,0)];
  }
  
}

void final_update_kernel(const int *dispx, const int *dispy, const int *dispz, const int *idx, const float *dt, const float *k1, const float *k2, const float* k3, float* k4, float *yy) {

  for (int i=0;i<6;i++) {
    k4[OPS_ACC_MD8(i,0,0,0)] = k4[OPS_ACC_MD8(i,0,0,0)]* *dt;
    yy[OPS_ACC_MD10(i,0,0,0)] = yy[OPS_ACC_MD9(i,0,0,0)] +
      k1[OPS_ACC_MD5(i,0,0,0)]/6.0f +
      k2[OPS_ACC_MD6(i,0,0,0)]/3.0f +
      k3[OPS_ACC_MD7(i,0,0,0)]/3.0f +
      k4[OPS_ACC_MD8(i,0,0,0)]/6.0f;
  }
  
}

void fd3d_pml_kernel(const int *dispx, const int *dispy, const int *dispz, const int *idx, const float *rho, const float *mu, const float* yy, float* dyy) {
  
// #include "../coeffs/coeffs8.h"
//  float* c = &coeffs[half+half*(order+1)];
  const float c[9] = {0.0035714285714285713,-0.0380952380952381,0.2,-0.8,0.0,0.8,-0.2,0.0380952380952381,-0.0035714285714285713};
  float invdx = 1.0 / dx;
  float invdy = 1.0 / dy;
  float invdz = 1.0 / dz;
  int xbeg=half;
  int xend=nx-half;
  int ybeg=half;
  int yend=ny-half;
  int zbeg=half;
  int zend=nz-half;
  int xpmlbeg=xbeg+pml_width;
  int ypmlbeg=ybeg+pml_width;
  int zpmlbeg=zbeg+pml_width;
  int xpmlend=xend-pml_width;
  int ypmlend=yend-pml_width;
  int zpmlend=zend-pml_width;

  float sigma = mu[OPS_ACC5(0,0,0)]/rho[OPS_ACC4(0,0,0)];
  float sigmax=0.0;
  float sigmay=0.0;
  float sigmaz=0.0;
  if(idx[0]<=xbeg+pml_width){
    sigmax = (xbeg+pml_width-idx[0])*sigma * 0.1f;///pml_width;
  }
  if(idx[0]>=xend-pml_width){
    sigmax=(idx[0]-(xend-pml_width))*sigma * 0.1f;///pml_width;
  }
  if(idx[1]<=ybeg+pml_width){
    sigmay=(ybeg+pml_width-idx[1])*sigma * 0.1f;///pml_width;
  }
  if(idx[1]>=yend-pml_width){
    sigmay=(idx[1]-(yend-pml_width))*sigma * 0.1f;///pml_width;
  }
  if(idx[2]<=zbeg+pml_width){
    sigmaz=(zbeg+pml_width-idx[2])*sigma * 0.1f;///pml_width;
  }
  if(idx[2]>=zend-pml_width){
    sigmaz=(idx[2]-(zend-pml_width))*sigma * 0.1f;///pml_width;
  }

					//sigmax=0.0;
					//sigmay=0.0;
  
  float px = yy[OPS_ACC_MD6(0,0,0,0)];
  float py = yy[OPS_ACC_MD6(1,0,0,0)];
  float pz = yy[OPS_ACC_MD6(2,0,0,0)];
  
  float vx = yy[OPS_ACC_MD6(3,0,0,0)];
  float vy = yy[OPS_ACC_MD6(4,0,0,0)];
  float vz = yy[OPS_ACC_MD6(5,0,0,0)];
  
  float vxx=0.0;
  float vxy=0.0;
  float vxz=0.0;
  
  float vyx=0.0;
  float vyy=0.0;
  float vyz=0.0;

  float vzx=0.0;
  float vzy=0.0;
  float vzz=0.0;
  
  float pxx=0.0;
  float pxy=0.0;
  float pxz=0.0;
  
  float pyx=0.0;
  float pyy=0.0;
  float pyz=0.0;

  float pzx=0.0;
  float pzy=0.0;
  float pzz=0.0;

  for(int i=-half;i<=half;i++){
    pxx += yy[OPS_ACC_MD6(0,i,0,0)]*c[i+half];
    pyx += yy[OPS_ACC_MD6(1,i,0,0)]*c[i+half];
    pzx += yy[OPS_ACC_MD6(2,i,0,0)]*c[i+half];
    
    vxx += yy[OPS_ACC_MD6(3,i,0,0)]*c[i+half];
    vyx += yy[OPS_ACC_MD6(4,i,0,0)]*c[i+half];
    vzx += yy[OPS_ACC_MD6(5,i,0,0)]*c[i+half];
    
    pxy += yy[OPS_ACC_MD6(0,0,i,0)]*c[i+half];
    pyy += yy[OPS_ACC_MD6(1,0,i,0)]*c[i+half];
    pzy += yy[OPS_ACC_MD6(2,0,i,0)]*c[i+half];
    
    vxy += yy[OPS_ACC_MD6(3,0,i,0)]*c[i+half];
    vyy += yy[OPS_ACC_MD6(4,0,i,0)]*c[i+half];
    vzy += yy[OPS_ACC_MD6(5,0,i,0)]*c[i+half];
    
    pxz += yy[OPS_ACC_MD6(0,0,0,i)]*c[i+half];
    pyz += yy[OPS_ACC_MD6(1,0,0,i)]*c[i+half];
    pzz += yy[OPS_ACC_MD6(2,0,0,i)]*c[i+half];
    
    vxz += yy[OPS_ACC_MD6(3,0,0,i)]*c[i+half];
    vyz += yy[OPS_ACC_MD6(4,0,0,i)]*c[i+half];
    vzz += yy[OPS_ACC_MD6(5,0,0,i)]*c[i+half];
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
  
  dyy[OPS_ACC_MD7(0,0,0,0)]=vxx/rho[OPS_ACC4(0,0,0)] - sigmax*px;
  dyy[OPS_ACC_MD7(3,0,0,0)]=(pxx+pyx+pxz)*mu[OPS_ACC5(0,0,0)] - sigmax*vx;
  
  dyy[OPS_ACC_MD7(1,0,0,0)]=vyy/rho[OPS_ACC4(0,0,0)] - sigmay*py;
  dyy[OPS_ACC_MD7(4,0,0,0)]=(pxy+pyy+pyz)*mu[OPS_ACC5(0,0,0)] - sigmay*vy;
  
  dyy[OPS_ACC_MD7(2,0,0,0)]=vzz/rho[OPS_ACC4(0,0,0)] - sigmaz*pz;
  dyy[OPS_ACC_MD7(5,0,0,0)]=(pxz+pyz+pzz)*mu[OPS_ACC5(0,0,0)] - sigmaz*vz;
  
}

void poisson_kernel_initialguess(float *u) {
  u[OPS_ACC0(0,0,0)] = 0.0;
}

void poisson_kernel_stencil(const float *u, const float *f, float *u2) {
  /* NEED TO PASS IN dx, dy, dz and above to compute this iteration correctly */
  u2[OPS_ACC2(0,0,0)] = (u[OPS_ACC0(-1,0,0)]+u[OPS_ACC0(1,0,0)])*0.125f
                      + (u[OPS_ACC0(0,-1,0)]+u[OPS_ACC0(0,1,0)])*0.125f
                      + (u[OPS_ACC0(0,0,-1)]+u[OPS_ACC0(0,0,1)])*0.125f
                      - f[OPS_ACC1(0,0,0)];
}

void poisson_kernel_update(const float *u2, float *u) {
  u[OPS_ACC1(0,0,0)] = u2[OPS_ACC0(0,0,0)];
}

void poisson_kernel_error(const float *u, const float *ref, float *err) {
  *err = *err + (u[OPS_ACC0(0,0,0)]-ref[OPS_ACC1(0,0,0)])*(u[OPS_ACC0(0,0,0)]-ref[OPS_ACC1(0,0,0)]);
  //printf("u,ref = %f %f\n",u[OPS_ACC0(0,0,0)],ref[OPS_ACC1(0,0,0)]);
}

#endif //poisson_KERNEL_H
