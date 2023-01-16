
#pragma once

#include <ops_seq_v2.h>
#include "blacksholes_common.h"

void ops_krnl_interior_init(ACC<float> & data, const int *idx, const float *deltaS, const float *strikePrice);

void ops_krnl_zero_init(ACC<float> &data);

void ops_krnl_const_init(ACC<float> &data, const float *constant);

void ops_krnl_copy(ACC<float> &data, const ACC<float>& data_new);

void ops_krnl_blacksholes(ACC<float> & current, const ACC<float> & next, ACC<float> & a, ACC<float> & b, ACC<float> & c,
		const float * alpha, const float * beta, const int * idx, const int * iter);

int bs_explicit1_ops(float* result, OPS_instance * ops_inst, GridParameter gridData, BlacksholesParameter computeParam);
