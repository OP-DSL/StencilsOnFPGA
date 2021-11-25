#include <ap_int.h>
#include <hls_stream.h>
#include <ap_axi_sdata.h>
#include <math.h>
#include <stdio.h>

//--vivado.prop run.impl_1.STEPS.vpl.update_bd.TCL.PRE=/home/kkvasan/vitis_ws/ddr4_tiled_RW/small_RW/src/stencil_axi.tcl
typedef ap_uint<512> uint512_dt;
typedef ap_uint<1024> uint1024_dt;
typedef ap_uint<256> uint256_dt;
typedef ap_uint<576> uint576_dt;
typedef ap_uint<288> uint288_dt;
typedef ap_axiu<256,0,0,0> t_pkt;
typedef ap_axiu<32,0,0,0> t_pkt_32;
typedef ap_axiu<1024,0,0,0> t_pkt_1024;

#define MAX_SIZE_X 768
#define MAX_DEPTH_16 (MAX_SIZE_X/16)
#define MAX_DEPTH_L (MAX_SIZE_X/64)
#define MAX_DEPTH_P (MAX_SIZE_X/64) * MAX_SIZE_X

//user function
#define PORT_WIDTH 18
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
const int max_depth_16 = MAX_DEPTH_L * 4;
const int max_depth_8 = MAX_DEPTH_L * 4;

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
	unsigned int plane_diff;
	unsigned int line_diff;
	unsigned int total_itr;
	unsigned int total_itr_R;
	unsigned int total_itr_W;
	unsigned short outer_loop_limit;
};

