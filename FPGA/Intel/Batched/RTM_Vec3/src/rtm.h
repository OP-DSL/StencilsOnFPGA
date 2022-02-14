#ifndef __RTM__
#define __RTM__
#define ORDER 4


// Host part

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


// Hardware Part
#define MAX_SIZE_X 128
#define MAX_DEPTH_16 (MAX_SIZE_X/16)

//user function
#define PORT_WIDTH 8
#define SHIFT_BITS 0
#define DIM 6
#define DATATYPE_SIZE 32
//#define BEAT_SHIFT_BITS 10
#define BURST_LEN MAX_DEPTH_16

#define STAGES 2
#define ORDER 4

const int max_size_y = MAX_SIZE_X;
const int min_size_y = 20;
const int avg_size_y = MAX_SIZE_X;

const int max_block_x = MAX_SIZE_X/1 + 1;
const int min_block_x = 20/1 + 1;
const int avg_block_x = MAX_SIZE_X/1 + 1;

const int max_grid = max_block_x * max_size_y * max_size_y;
const int min_grid = min_block_x * min_size_y * min_size_y;
const int avg_grid = avg_block_x * avg_size_y * avg_size_y;

const int max_grid_2 = (max_block_x * max_size_y * max_size_y)/2;
const int min_grid_2 = (min_block_x * min_size_y * min_size_y)/2;
const int avg_grid_2 = (avg_block_x * avg_size_y * avg_size_y)/2;

const int port_width  = PORT_WIDTH;
const int max_depth_16 = MAX_DEPTH_16 * 8;
const int max_depth_8 = MAX_DEPTH_16 * 8;

const int plane_buff_size = ((48+2)/3)*48;
const int line_buff_size = (48+2)/3;


typedef union  {
   int i;
   float f;
} data_conv;

struct data_G{
	unsigned short sizex;
	unsigned short sizey;
	unsigned short sizez;
	unsigned short grid_sizex;
	unsigned short grid_sizey;
	unsigned short grid_sizez;
	unsigned short limit_z;
	unsigned short xblocks;
	unsigned int gridsize_pr;
	unsigned int plane_diff;
	unsigned int plane_size;
	unsigned int line_diff;
	unsigned short outer_loop_limit;
};



#define INC0(x) ((x))
#define INC1(x) ((x+8))
#define INC2(x) ((x+16))

#endif
