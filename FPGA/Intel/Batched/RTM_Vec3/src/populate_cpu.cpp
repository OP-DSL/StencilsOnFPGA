#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "rtm.h"



int copy_ToVec(float* grid_s, IntVector &grid_d1, IntVector &grid_d2, IntVector &grid_d3, int grid_size, int delay){
  printf("grid_size:%d\n", grid_size);
  printf("Vector size %d %d %d\n", grid_d1.size(), grid_d1.size(), grid_d1.size());
  printf("loop limit %d\n", grid_size/(16*sizeof(float)));
  for(int i = 0; i < grid_size/(16*sizeof(float)); i++){
      for(int v = 0; v < 16; v++){
        if((i % 3) == 0){
          grid_d1[i/3+delay].data[v] = grid_s[i*16+v];
        } else if ((i % 3) == 1){
          grid_d2[i/3+delay].data[v] = grid_s[i*16+v];
        }else if ((i % 3) == 2){
          grid_d3[i/3+delay].data[v] = grid_s[i*16+v];
        }
      }

  }
    return 0;
}

int copy_FromVec(IntVector &grid_d1, IntVector &grid_d2,  IntVector &grid_d3, float* grid_s, int grid_size, int delay){
  printf("grid_size:%d\n", grid_size);
  for(int i = 0; i < grid_size/(16*sizeof(float)); i++){
      for(int v = 0; v < 16; v++){
        if((i % 3) == 0){
          grid_s[i*16+v] = grid_d1[i/3+delay].data[v];
        } else if ((i % 3) == 1) {
          grid_s[i*16+v] = grid_d2[i/3+delay].data[v];
        } else if ((i % 3) == 2) {
          grid_s[i*16+v] = grid_d3[i/3+delay].data[v];
        }
      }

  }
    return 0;
}

int populate_rho_mu_yy(float* grid, struct Grid_d grid_d){
  for(int i = 0; i < grid_d.act_sizez; i++){
    for(int j = 0; j < grid_d.act_sizey; j++){
      for(int k = 0; k < grid_d.act_sizex; k++){

        if(i < ORDER || i >= (grid_d.logical_size_z+ORDER) || j < ORDER || j >= (grid_d.logical_size_y+ORDER) || k < ORDER || k >= (grid_d.logical_size_x+ORDER)){
          for(int p = 0; p < 8; p++){
            grid[i * grid_d.grid_size_x * grid_d.grid_size_y * 8 + j * grid_d.grid_size_x * 8 + k*8 + p] = 0;
          }
        } else {
          float x = 1.0 * (float ) (k - grid_d.act_sizex/2)/(grid_d.logical_size_x);
          float y = 1.0 * (float ) (j - grid_d.act_sizey/2)/(grid_d.logical_size_y);
          float z = 1.0 * (float ) (i - grid_d.act_sizez/2)/(grid_d.logical_size_z);

          const float C = 1;
          const float r0 = 0.1;

          grid[i * grid_d.grid_size_x * grid_d.grid_size_y * 8 + j * grid_d.grid_size_x * 8 + k*8 + 0] = 1000.0f;
          grid[i * grid_d.grid_size_x * grid_d.grid_size_y * 8 + j * grid_d.grid_size_x * 8 + k*8 + 1] = 0.001f;
          grid[i * grid_d.grid_size_x * grid_d.grid_size_y * 8 + j * grid_d.grid_size_x * 8 + k*8 + 2] = (1.0/3) * C * exp(-(x*x+y*y+z*z)/r0);
          for(int p = 3; p < 8; p++){
            grid[i * grid_d.grid_size_x * grid_d.grid_size_y * 8 + j * grid_d.grid_size_x * 8 + k*8 + p] = 0.0;
          }
        }
      }
    }
  }
  return 0;
}

int calc_ytemp_kernel(float* rho_mu_yy, float* k_grid, float dt, float* rho_mu_yy_temp, float val, struct Grid_d grid_d){
  for(int i = ORDER; i < grid_d.act_sizez - ORDER; i++){
    for(int j = ORDER; j < grid_d.act_sizey - ORDER; j++){
      for(int k = ORDER; k < grid_d.act_sizex - ORDER; k++){
        int index_b = i * grid_d.grid_size_x * grid_d.grid_size_y * 8 + j * grid_d.grid_size_x * 8 + k*8;
        rho_mu_yy_temp[index_b] = rho_mu_yy[index_b];
        rho_mu_yy_temp[index_b+1] = rho_mu_yy[index_b+1];
        for(int p = 2; p < 8; p++){
          int index = i * grid_d.grid_size_x * grid_d.grid_size_y * 8 + j * grid_d.grid_size_x * 8 + k*8 + p;
          k_grid[index] *= dt;
          rho_mu_yy_temp[index] = rho_mu_yy[index] + k_grid[index] * val;
        }      
      }
    }
  }  
  return 0;
}


int final_update_kernel(float* rho_mu_yy, float* k_grid1, float* k_grid2, float* k_grid3, float* k_grid4, float dt, struct Grid_d grid_d){
  for(int i = ORDER; i < grid_d.act_sizez - ORDER; i++){
    for(int j = ORDER; j < grid_d.act_sizey - ORDER; j++){
      for(int k = ORDER; k < grid_d.act_sizex - ORDER; k++){
        int index_b = i * grid_d.grid_size_x * grid_d.grid_size_y * 8 + j * grid_d.grid_size_x * 8 + k*8;
        for(int p = 2; p < 8; p++){
          int index = i * grid_d.grid_size_x * grid_d.grid_size_y * 8 + j * grid_d.grid_size_x * 8 + k*8 + p;
          k_grid4[index] *= dt;
          rho_mu_yy[index] = rho_mu_yy[index] + k_grid1[index] * 0.1666666667f+ k_grid2[index] * 0.33333333333f + k_grid3[index] * 0.33333333333f + k_grid4[index]*0.1666666667f;
        }      
      }
    }
  }  
  return 0;
}


double square_error(float* current, float* next, struct Grid_d grid_d){
    double sum = 0;
    double sq_sum = 0;
    for(int i = 0; i < grid_d.grid_size_z; i++){
      for(int j = 0; j < grid_d.grid_size_y; j++){
        for(int k = 0; k < grid_d.grid_size_x; k++){
          for(int p = 2; p < 8; p++){
            float val1 = next[i*grid_d.grid_size_x*grid_d.grid_size_y*8 + j*grid_d.grid_size_x*8 + k*8 + p];
            float val2 = current[i*grid_d.grid_size_x*grid_d.grid_size_y*8 + j*grid_d.grid_size_x*8 + k*8 + p];
            double sq_error = (val1-val2) * (val1-val2);
            double s_sum = (val1+val2) * (val1+val2);
            double err_ratio = fabs((val1-val2)/val2);
            if((!isunordered(err_ratio,1.0) && err_ratio > 0.01 && fabs(val2) > 0.000001)){
            	printf("(%d %d %d %d %f %f) \n", i,j,k,p, val1 , val2);
            }
            sum += sq_error;
            if(!isunordered(sq_error/s_sum,1.0))
            	sq_sum += sq_error/s_sum ;
        }
//          printf("\n");
        }
//        printf("\n");
      }
    }
    printf("sum of sq_error/sq_sum ratio is %f \n", sq_sum );
    return sum;
}


int dump_rho_mu_yy(float* grid, struct Grid_d grid_d, char* n_rho, char* n_mu, char* n_yy){
  FILE* fp_rho = fopen(n_rho, "w");
  FILE* fp_mu  = fopen(n_mu, "w");
  FILE* fp_yy  = fopen(n_yy, "w");

  for(int i = 0; i < grid_d.act_sizez; i++){
    for(int j = 0; j < grid_d.act_sizey; j++){
      for(int k = 0; k < grid_d.act_sizex; k++){
        fprintf(fp_rho, "%e ", grid[i * grid_d.grid_size_x * grid_d.grid_size_y * 8 + j * grid_d.grid_size_x * 8 + k*8 + 0]);
        fprintf(fp_mu, "%e ",  grid[i * grid_d.grid_size_x * grid_d.grid_size_y * 8 + j * grid_d.grid_size_x * 8 + k*8 + 1]);
        for(int p = 2; p < 8; p++){
            fprintf(fp_yy, "%e ",  grid[i * grid_d.grid_size_x * grid_d.grid_size_y * 8 + j * grid_d.grid_size_x * 8 + k*8 + p]);
        }
      }
      fprintf(fp_rho, "\n");
      fprintf(fp_mu, "\n");
      fprintf(fp_yy, "\n");
    }
    fprintf(fp_rho, "\n");
    fprintf(fp_mu, "\n");
    fprintf(fp_yy, "\n");
  }
  fclose(fp_rho);
  fclose(fp_mu);
  fclose(fp_yy);
  return 0;
}


inline int caculate_index(struct Grid_d grid_d, int z, int y, int x, int pt){
  return (z * grid_d.grid_size_x * grid_d.grid_size_y * 8 + y * grid_d.grid_size_x * 8 + x*8 + pt);
}



void fd3d_pml_kernel(float* yy, float* dyy, struct Grid_d grid_d){
  int half = 4;
  int pml_width = 10;
  double dx = 0.005;
  double dy = 0.005;
  double dz = 0.005;

  float coeffs[9][9] = {
    {-2.717857142857143,8.0,-14.0,18.666666666666668,-17.5,11.2,-4.666666666666667,1.1428571428571428,-0.125}, 
    {-0.125,-1.5928571428571427,3.5,-3.5,2.9166666666666665,-1.75,0.7,-0.16666666666666666,0.017857142857142856}, 
    {0.017857142857142856,-0.2857142857142857,-0.95,2.0,-1.25,0.6666666666666666,-0.25,0.05714285714285714,-0.005952380952380952},
    {-0.005952380952380952,0.07142857142857142,-0.5,-0.45,1.25,-0.5,0.16666666666666666,-0.03571428571428571,0.0035714285714285713}, 
    {0.0035714285714285713,-0.0380952380952381,0.2,-0.8,0.0,0.8,-0.2,0.0380952380952381,-0.0035714285714285713}, 
    {-0.0035714285714285713,0.03571428571428571,-0.16666666666666666,0.5,-1.25,0.45,0.5,-0.07142857142857142,0.005952380952380952}, 
    {0.005952380952380952,-0.05714285714285714,0.25,-0.6666666666666666,1.25,-2.0,0.95,0.2857142857142857,-0.017857142857142856},
    {-0.017857142857142856,0.16666666666666666,-0.7,1.75,-2.9166666666666665,3.5,-3.5,1.5928571428571427,0.125}, 
    {0.125,-1.1428571428571428,4.666666666666667,-11.2,17.5,-18.666666666666668,14.0,-8.0,2.717857142857143}
  };


  float* c = &coeffs[half][half];
  float invdx = 1.0 / dx;
  float invdy = 1.0 / dy;
  float invdz = 1.0 / dz;
  int xbeg=half;
  int xend=grid_d.logical_size_x -half;
  int ybeg=half;
  int yend=grid_d.logical_size_y-half;
  int zbeg=half;
  int zend=grid_d.logical_size_z-half;
  int xpmlbeg=xbeg+pml_width;
  int ypmlbeg=ybeg+pml_width;
  int zpmlbeg=zbeg+pml_width;
  int xpmlend=xend-pml_width;
  int ypmlend=yend-pml_width;
  int zpmlend=zend-pml_width;



  for(int i = ORDER; i < grid_d.logical_size_z+ORDER; i++){
    for(int j = ORDER; j < grid_d.logical_size_y+ORDER; j++){
      for(int k = ORDER; k < grid_d.logical_size_x+ORDER; k++){


        float sigma = yy[caculate_index(grid_d,i,j,k,1)] / yy[caculate_index(grid_d,i,j,k,0)];
        float sigmax=0.0;
        float sigmay=0.0;
        float sigmaz=0.0;

        int x = k - ORDER;
        int y = j - ORDER;
        int z = i - ORDER;

        if(x<=xbeg+pml_width){
          sigmax = (xbeg+pml_width-x)*sigma/pml_width;
        }
        if( x >= xend-pml_width){
          sigmax=( x -(xend-pml_width))*sigma/pml_width;
        }
        if(y <=ybeg+pml_width){
          sigmay=(ybeg+pml_width-y)*sigma/pml_width;
        }
        if(y >=yend-pml_width){
          sigmay=(y-(yend-pml_width))*sigma/pml_width;
        }
        if(z <=zbeg+pml_width){
          sigmaz=(zbeg+pml_width- z)*sigma/pml_width;
        }
        if( z >=zend-pml_width){
          sigmaz=( z -(zend-pml_width))*sigma/pml_width;
        }

        //sigmax=0.0;
        //sigmay=0.0;
        
        float px = yy[caculate_index(grid_d,i,j,k,2)];
        float py = yy[caculate_index(grid_d,i,j,k,3)];
        float pz = yy[caculate_index(grid_d,i,j,k,4)];
        
        float vx = yy[caculate_index(grid_d,i,j,k,5)];
        float vy = yy[caculate_index(grid_d,i,j,k,6)];
        float vz = yy[caculate_index(grid_d,i,j,k,7)];
        
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

        for(int l=-half;l<=half;l++){
          pxx += yy[caculate_index(grid_d,i,j,k+l,2)]*c[l];
          pyx += yy[caculate_index(grid_d,i,j,k+l,3)]*c[l];
          pzx += yy[caculate_index(grid_d,i,j,k+l,4)]*c[l];
          
          vxx += yy[caculate_index(grid_d,i,j,k+l,5)]*c[l];
          vyx += yy[caculate_index(grid_d,i,j,k+l,6)]*c[l];
          vzx += yy[caculate_index(grid_d,i,j,k+l,7)]*c[l];
          
          pxy += yy[caculate_index(grid_d,i,j+l,k,2)]*c[l];
          pyy += yy[caculate_index(grid_d,i,j+l,k,3)]*c[l];
          pzy += yy[caculate_index(grid_d,i,j+l,k,4)]*c[l];
          
          vxy += yy[caculate_index(grid_d,i,j+l,k,5)]*c[l];
          vyy += yy[caculate_index(grid_d,i,j+l,k,6)]*c[l];
          vzy += yy[caculate_index(grid_d,i,j+l,k,7)]*c[l];
          
          pxz += yy[caculate_index(grid_d,i+l,j,k,2)]*c[l];
          pyz += yy[caculate_index(grid_d,i+l,j,k,3)]*c[l];
          pzz += yy[caculate_index(grid_d,i+l,j,k,4)]*c[l];
          
          vxz += yy[caculate_index(grid_d,i+l,j,k,5)]*c[l];
          vyz += yy[caculate_index(grid_d,i+l,j,k,6)]*c[l];
          vzz += yy[caculate_index(grid_d,i+l,j,k,7)]*c[l];
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
      
        dyy[caculate_index(grid_d,i,j,k,0)] = yy[caculate_index(grid_d,i,j,k,0)];
        dyy[caculate_index(grid_d,i,j,k,1)] = yy[caculate_index(grid_d,i,j,k,1)];

        dyy[caculate_index(grid_d,i,j,k,2)]= vxx/yy[caculate_index(grid_d,i,j,k,0)]- sigmax*px;
        dyy[caculate_index(grid_d,i,j,k,5)]= (pxx+pyx+pxz)*yy[caculate_index(grid_d,i,j,k,1)] - sigmax*vx;
        
        dyy[caculate_index(grid_d,i,j,k,3)]= vyy/yy[caculate_index(grid_d,i,j,k,0)]  - sigmay*py;
        dyy[caculate_index(grid_d,i,j,k,6)]= (pxy+pyy+pyz)*yy[caculate_index(grid_d,i,j,k,1)]  - sigmay*vy;
        
        dyy[caculate_index(grid_d,i,j,k,4)]= vzz/yy[caculate_index(grid_d,i,j,k,0)]  - sigmaz*pz;
        dyy[caculate_index(grid_d,i,j,k,7)]= (pxz+pyz+pzz)*yy[caculate_index(grid_d,i,j,k,1)]  - sigmaz*vz;


      }
    }
  }
}
