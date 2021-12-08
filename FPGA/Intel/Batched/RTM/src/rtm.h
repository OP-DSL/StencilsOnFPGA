#ifndef __RTM__
#define __RTM__
#define ORDER 4


struct Grid_d
{
	int logical_size_x;
	int logical_size_y;
	int logical_size_z;
	int act_sizex;
	int act_sizey;
	int act_sizez;
	int grid_size_x;
	int grid_size_y;
	int grid_size_z;
	int order;
	int dims;
	int data_size_bytes_dim1;
	int data_size_bytes_dim6;
	int data_size_bytes_dim8;

};

int populate_rho_mu_yy(float* grid, struct Grid_d grid_d);
int calc_ytemp_kernel(float* rho_mu_yy, float* k, float dt, float* rho_mu_yy_temp, float val, struct Grid_d grid_d);
int dump_rho_mu_yy(float* grid, struct Grid_d grid_d);
int dump_rho_mu_yy(float* grid, struct Grid_d grid_d, char* n_rho, char* n_mu, char* n_yy);
double square_error(float* current, float* next, struct Grid_d grid_d);

int copy_grid(float* grid_s, float* grid_d, int grid_size);
inline int caculate_index(struct Grid_d grid_d, int z, int y, int x, int pt);
void fd3d_pml_kernel(float* yy, float* dyy, struct Grid_d grid_d);
int final_update_kernel(float* rho_mu_yy, float* k_grid1, float* k_grid2, float* k_grid3, float* k_grid4, float dt, struct Grid_d grid_d);

#endif
