/** @author Beniel Thileepan
  * 
  */

#pragma once

#include <cmath>
#include <iostream>
#include <algorithm>
#include <math.h>
#include <vector>
#include <assert.h>
#include <chrono>
#include "heat3D_common.h"

int heat3D_explicit(float * current, float *next, GridParameter gridData, std::vector<heat3DParameter> & calcParam);

void initialize_grid(float* grid, GridParameter gridData);

void copy_grid(float* grid_s, float* grid_d, GridParameter gridData);
