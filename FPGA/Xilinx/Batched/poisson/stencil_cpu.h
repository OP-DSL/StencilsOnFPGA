#include <stdlib.h>
#include <stdio.h>

int stencil_computation(float* current, float* next, struct Grid_Parameter data_g);
double square_error(float* current, float* next, struct Grid_Parameter data_g);
int copy_grid(float* grid_s, float* grid_d, struct Grid_Parameter data_g);
int initialise_grid(float* grid, struct Grid_Parameter data_g);

struct Grid_Parameter{

	unsigned int logical_size_x;
	unsigned int logical_size_y;

	unsigned int act_size_x;
	unsigned int act_size_y;

	unsigned int grid_size_x;
	unsigned int grid_size_y;

	unsigned int batch;

};

