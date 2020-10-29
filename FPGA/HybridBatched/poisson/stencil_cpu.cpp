#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "stencil_cpu.h"


// golden non optimised stencil computation on host PC
int stencil_computation(float* current, float* next, struct Grid_Parameter data_g){
	for(int bat = 0; bat < data_g.batch; bat++){
		int offset = bat * data_g.grid_size_x* data_g.grid_size_y;
		for(int i = 0; i < data_g.act_size_y; i++){
		  for(int j = 0; j < data_g.grid_size_x; j = j+8){
			if(i == 0 || j == 0 || i == data_g.act_size_y -1  || j==(data_g.act_size_x-1)*8){
				for(int v = 0; v < 8;v++){
						next[i*data_g.grid_size_x + j+v + offset] = current[i*data_g.grid_size_x + j+v + offset] ;
				}
			} else {
				for(int v = 0; v < 8;v++){
				  next[i*data_g.grid_size_x + j+v + offset] = current[i*data_g.grid_size_x + j+v + offset] * 0.5f + \
											   (current[(i-1)*data_g.grid_size_x + j+v + offset] + current[(i+1)*data_g.grid_size_x + j+v +offset]) * 0.125f + \
											   (current[i*data_g.grid_size_x + j+v+8+offset] + current[i*data_g.grid_size_x + j+v-8+offset]) * 0.125f;
				}
			}
		  }
		}
	}
    return 0;
}

// function to compare difference of two grids
double square_error(float* current, float* next, struct Grid_Parameter data_g){
  double sum = 0;
  for(int bat = 0; bat < data_g.batch; bat++){
	int offset = bat * data_g.grid_size_x* data_g.grid_size_y;
    for(int i = 0; i < data_g.act_size_y; i++){
      for(int j = 0; j < data_g.grid_size_x; j++){
    	int index = i*data_g.grid_size_x + j+offset;
    	float v1 = (next[index]);
    	float v2 = (current[index]);
    	if(fabs(v1-v2)/(fabs(v1) + fabs(v2)) >= 0.000001 && (fabs(v1) + fabs(v2)) > 0.000001 ){
    		printf("i:%d j:%d v1:%f v2:%f\n", v1, v2, i, j);
    	}

        sum += next[index]*next[index] - current[index]*current[index];
      }
    }
  }
    return sum;
}

int copy_grid(float* grid_s, float* grid_d, struct Grid_Parameter data_g){
    for(int bat = 0; bat < data_g.batch; bat++){
    	int offset = bat * data_g.grid_size_x* data_g.grid_size_y;
		for(int i = 0; i < data_g.act_size_y; i++){
		  for(int j = 0; j < data_g.grid_size_x; j++){
			grid_d[i*data_g.grid_size_x + j+offset] = grid_s[i*data_g.grid_size_x + j+offset];
		  }
		}
    }
    return 0;
}


// function to set boundary values
int initialise_grid(float* grid, struct Grid_Parameter data_g){
	for(int bat = 0; bat < data_g.batch; bat++){
	int offset = bat * data_g.grid_size_x* data_g.grid_size_y;
	  for(int i = 0; i < data_g.act_size_y; i++){
	  for(int j = 0; j < data_g.grid_size_x; j = j + 8){
		  if(i == 0 || j == 0 || i == data_g.act_size_y -1  || j==(data_g.act_size_x-1)*8){
			  for(int v = 0; v < 8;v++){
				float r = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
				grid[i*data_g.grid_size_x + j+v +offset] = r;
			  }
		  } else {
			  for(int v = 0; v < 8;v++){
				  grid[i*data_g.grid_size_x + j+v +offset] = 0;
			  }
		  }
		}
	  }
	}
  return 0;
}
