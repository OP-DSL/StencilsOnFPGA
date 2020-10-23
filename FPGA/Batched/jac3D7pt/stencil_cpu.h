#include <stdio.h>
#ifndef __STENCIL_CPU__
#define __STENCIL_CPU__

struct Grid_Parameter{

	unsigned int logical_size_x;
	unsigned int logical_size_y;
	unsigned int logical_size_z;

	unsigned int act_size_x;
	unsigned int act_size_y;
	unsigned int act_size_z;

	unsigned int grid_size_x;
	unsigned int grid_size_y;
	unsigned int grid_size_z;

	unsigned int batch;

};

inline int index_cal(int i,int j,int k, struct Grid_Parameter data_g);
int stencil_computation(float* current, float* next, struct Grid_Parameter data_g);
double square_error(float* current, float* next, struct Grid_Parameter data_g);
int copy_grid(float* grid_s, float* grid_d, unsigned int grid_size);
int initialise_grid(float* grid, struct Grid_Parameter data_g);

#endif
