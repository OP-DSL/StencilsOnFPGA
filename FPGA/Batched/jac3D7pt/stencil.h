#include <ap_int.h>
#include <hls_stream.h>
#include <ap_axi_sdata.h>
#include <math.h>
#include <stdio.h>


typedef ap_uint<512> uint512_dt;
typedef ap_uint<256> uint256_dt;
typedef ap_axiu<256,0,0,0> t_pkt;
typedef ap_axiu<32,0,0,0> t_pkt_32;

#define MAX_SIZE_X 304
#define MAX_DEPTH_16 (MAX_SIZE_X/16)

//user function
#define PORT_WIDTH 8
#define SHIFT_BITS 3
#define DATATYPE_SIZE 32
//#define BEAT_SHIFT_BITS 10
#define BURST_LEN MAX_DEPTH_16

#define STAGES 2

const int max_size_y = MAX_SIZE_X;
const int min_size_y = 20;
const int avg_size_y = MAX_SIZE_X;

const int max_block_x = MAX_SIZE_X/8 + 1;
const int min_block_x = 20/8 + 1;
const int avg_block_x = MAX_SIZE_X/8 + 1;

const int max_grid = max_block_x * max_size_y * max_size_y;
const int min_grid = min_block_x * min_size_y * min_size_y;
const int avg_grid = avg_block_x * avg_size_y * avg_size_y;

const int max_grid_2 = (max_block_x * max_size_y * max_size_y)/2;
const int min_grid_2 = (min_block_x * min_size_y * min_size_y)/2;
const int avg_grid_2 = (avg_block_x * avg_size_y * avg_size_y)/2;

const int port_width  = PORT_WIDTH;
const int max_depth_16 = MAX_DEPTH_16 * 4;
const int max_depth_8 = MAX_DEPTH_16 * 4;
const int max_depth_xy = max_block_x * MAX_SIZE_X;

typedef union  {
   int i;
   float f;
} data_conv;

struct data_G{
	unsigned short sizex;
	unsigned short sizey;
	unsigned short sizez;
	unsigned short xdim;
	unsigned short xblocks;
	unsigned short grid_sizey;
	unsigned short grid_sizez;
	unsigned short limit_z;
	unsigned short offset_x;
	unsigned short tile_x;
	unsigned short offset_y;
	unsigned short tile_y;
	unsigned int plane_size;
	unsigned int gridsize_pr;
	unsigned int gridsize_da;
	unsigned int plane_diff;
	unsigned int line_diff;
	unsigned short outer_loop_limit;
	unsigned int total_itr;
	unsigned short batches;
};

