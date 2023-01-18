
#pragma once

#include <ops_seq_v2.h>
#include "blacksholes_common.h"
#include "blacksholes_ops_kernels.h"

int bs_explicit1_ops(float* result, OPS_instance * ops_inst, GridParameter gridData, BlacksholesParameter computeParam);