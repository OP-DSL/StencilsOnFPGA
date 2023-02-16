#include <ap_int.h>
#include <hls_stream.h>
#include <ap_axi_sdata.h>
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include "heat3D_common.h"

#pragma once

typedef ap_uint<512> uint512_dt;
typedef ap_uint<256> uint256_dt;
typedef ap_axiu<256,0,0,0> t_pkt;
typedef ap_axiu<32,0,0,0> t_pkt_32;

#define SLR_P_STAGE NUM_OF_PROCESS_GRID_PER_SLR

//Maximum Tile Size
#define MAX_SIZE_X 304
#define MAX_DEPTH_16 (MAX_SIZE_X/16)

//user function
#define VEC_FACTOR 8
#define SHIFT_BITS 3
#define DATATYPE_SIZE 32  // single precision operations


const int max_size_y = MAX_SIZE_X;
const int min_size_y = 20;
const int avg_size_y = MAX_SIZE_X;

const int max_block_x = MAX_SIZE_X/VEC_FACTOR + 1;
const int min_block_x = 20/VEC_FACTOR + 1;
const int avg_block_x = MAX_SIZE_X/VEC_FACTOR + 1;

const int max_grid = max_block_x * max_size_y * max_size_y;
const int min_grid = min_block_x * min_size_y * min_size_y;
const int avg_grid = avg_block_x * avg_size_y * avg_size_y;

const int vec_factor = VEC_FACTOR;
const int max_depth_16 = MAX_DEPTH_16;
const int max_depth_8 = MAX_DEPTH_16 *2;
const int max_depth_xy = max_block_x * max_size_y;

// union to reinterpret float as integer and vice versa
typedef union  {
   int i;
   float f;
} data_conv;


// strcutre to hold grid parameters to avoid recalculation in
// different process
//struct data_G{
//	unsigned short sizex;
//	unsigned short sizey;
//	unsigned short xdim0;
//	unsigned short end_index;
//	unsigned short end_row;
//	unsigned int gridsize;
//    unsigned int total_itr_512;
//    unsigned int total_itr_256;
//	unsigned short outer_loop_limit;
//	unsigned short endrow_plus2;
//	unsigned short endrow_plus1;
//	unsigned short endrow_minus1;
//	unsigned short endindex_minus1;
//};

struct data_G{
	unsigned short sizex;
	unsigned short sizey;
	unsigned short sizez;
	unsigned short xblocks;
	unsigned short grid_size_x;
	unsigned short grid_size_y;
	unsigned short grid_size_z;
	unsigned short limit_z;
	unsigned short offset_x;
	unsigned short offset_y;
	unsigned short offset_z;
	unsigned int plane_size;
	unsigned int gridsize_pr;
	unsigned int gridsize_da;
	unsigned int plane_diff;
	unsigned int line_diff;
	unsigned short outer_loop_limit;
	unsigned int total_itr;
	bool last_half;
	unsigned short batches;
};

