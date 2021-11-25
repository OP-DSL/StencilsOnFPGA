#include "stencil_cpu.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

inline int index_cal(int i,int j,int k, struct Grid_Parameter data_g) {
	return (i*data_g.grid_size_x*data_g.grid_size_y + j * data_g.grid_size_x + k);
}

int stencil_computation(float* current, float* next, struct Grid_Parameter data_g){
	for(int b = 0; b < data_g.batch; b++){
		unsigned int offset_b = b* data_g.grid_size_x * data_g.grid_size_y * data_g.grid_size_z;
		for(int i = 0; i < data_g.act_size_z; i++){
		  for(int j = 0; j < data_g.act_size_y; j++){
			for(int k = 0; k < data_g.act_size_x; k++){
			  if(i == 0 || j == 0 || k ==0 || i == data_g.act_size_z -1  || j==data_g.act_size_y-1 || k == data_g.act_size_x -1){
				next[offset_b+ index_cal(i,j,k,data_g)] = current[offset_b + index_cal(i,j,k,data_g)] ;
			  } else {
				next[offset_b+ index_cal(i,j,k,data_g)] = current[offset_b + index_cal(i-1,j,k,data_g)] * (0.01f)  + \
											  	  	  	  current[offset_b + index_cal(i+1,j,k,data_g)] * (0.02f)  + \
														  current[offset_b + index_cal(i,j-1,k,data_g)] * (0.03f)  + \
														  current[offset_b + index_cal(i,j+1,k,data_g)] * (0.04f)  + \
														  current[offset_b + index_cal(i,j,k-1,data_g)] * (0.05f)  + \
														  current[offset_b + index_cal(i,j,k+1,data_g)] * (0.06f)  + \
														  current[offset_b + index_cal(i,j,k,data_g)] * (0.79) ;
			  }
			}
		  }
		}
	}
    return 0;
}

double square_error(float* current, float* next, struct Grid_Parameter data_g){
    double sum = 0;
    int count = 0;
    for(int b = 0; b < data_g.batch; b++){
    	unsigned int offset_b = b* data_g.grid_size_x * data_g.grid_size_y * data_g.grid_size_z;
		for(int i = 0; i < data_g.act_size_z; i++){
		  for(int j = 0; j < data_g.act_size_y; j++){
			for(int k = 0; k < data_g.act_size_x; k++){
			  unsigned int index = offset_b + index_cal(i,j,k,data_g);
			  float val1 = next[index];
			  float val2 = current[index];
			  sum +=  val1*val1 - val2*val2;
			  if((fabs(val1 -val2)/(fabs(val1) + fabs(val2))) > 0.0001 && (fabs(val1) + fabs(val2)) > 0.000001){
				  printf("z:%d y:%d x:%d Val1:%f val2:%f\n", i, j, k, val1, val2);
				  count++;
			  }
			}
		  }
		}
    }
    printf("Error count is %d\n", count);
    return sum;
}

int copy_grid(float* grid_s, float* grid_d, unsigned int grid_size){
    memcpy(grid_d, grid_s, grid_size);
    return 0;
}


int initialise_grid(float* grid, struct Grid_Parameter data_g){
	for(int b =0; b < data_g.batch; b++){
	unsigned int offset_b = b* data_g.grid_size_x * data_g.grid_size_y * data_g.grid_size_z;
	  for(int i = 0; i < data_g.act_size_z; i++){
		for(int j = 0; j < data_g.act_size_y; j++){
		  for(int k = 0; k < data_g.act_size_x; k++){
	        if(i == 0 || j == 0 || k == 0 || i == data_g.act_size_z -1  || j==data_g.act_size_y-1 || k == data_g.act_size_x-1 ){
			  float r = (static_cast <float> (RAND_MAX)-static_cast <float> (rand())) / static_cast <float> (RAND_MAX);
			  grid[offset_b + index_cal(i,j,k,data_g)] = r;
	        } else {
	          grid[offset_b + index_cal(i,j,k,data_g)] = 0;
	        }
		  }
		}
	  }
	}
  return 0;
}

