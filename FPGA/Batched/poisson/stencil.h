#include <ap_int.h>
#include <hls_stream.h>
#include <ap_axi_sdata.h>
#include <math.h>

#ifndef __STENCIL_HEADER__
#define __STENCIL_HEADER__

typedef ap_uint<512> uint512_dt;
typedef ap_uint<256> uint256_dt;
typedef ap_axiu<256,0,0,0> t_pkt;


#define SLR0_P_STAGE 20
#define SLR1_P_STAGE 20
#define SLR2_P_STAGE 20

//Maximum Tile Size
#define MAX_SIZE_X 8192
#define MAX_DEPTH_16 (MAX_SIZE_X/16)

//user function
#define VEC_FACTOR 8
#define SHIFT_BITS 3
#define DATATYPE_SIZE 32  // single precesion operations


const int max_size_y = MAX_SIZE_X;
const int min_size_y = 20;
const int avg_size_y = MAX_SIZE_X;

const int max_block_x = MAX_SIZE_X/16 + 1;
const int min_block_x = 20/16 + 1;
const int avg_block_x = MAX_SIZE_X/16 + 1;

const int max_grid = max_block_x * max_size_y;
const int min_grid = min_block_x * min_size_y;
const int avg_grid = avg_block_x * avg_size_y;

const int vec_factor  = VEC_FACTOR;
const int max_depth_16 = MAX_DEPTH_16;
const int max_depth_8 = MAX_DEPTH_16*2;

// union to reinterpret float as integer and vice versa
typedef union  {
   int i;
   float f;
} data_conv;


// strcutre to hold grid parameters to avoid recalculation in
// different process
struct data_G{
	unsigned short sizex;
	unsigned short sizey;
	unsigned short xdim0;
	unsigned short end_index;
	unsigned short end_row;
	unsigned int gridsize;
    unsigned int total_itr_512;
    unsigned int total_itr_256;
	unsigned short outer_loop_limit;
	unsigned short endrow_plus2;
	unsigned short endrow_plus1;
	unsigned short endrow_minus1;
	unsigned short endindex_minus1;
};

#endif
