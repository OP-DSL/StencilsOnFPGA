
#include <iostream>
#include <cstdlib>
#include "xcl2.hpp"
#include "blacksholes_cpu.h"
/******************************************************************************
* Main program
*******************************************************************************/
int main(int argc, char **argv)
{
	GridParameter gridProp;
	gridProp.logical_size_x = 200;
	gridProp.batch = 2;
	gridProp.num_iter = 2000;

	unsigned int vectorization_factor = 8;

	// setting grid parameters given by user
	const char* pch;

	for ( int n = 1; n < argc; n++ )
	{
		pch = strstr(argv[n], "-size=");

		if(pch != NULL)
		{
			gridProp.logical_size_x = atoi ( argv[n] + 7 ); continue;
		}

		pch = strstr(argv[n], "-iters=");

		if(pch != NULL)
		{
			gridProp.num_iter = atoi ( argv[n] + 7 ); continue;
		}
		pch = strstr(argv[n], "-batch=");

		if(pch != NULL)
		{
			gridProp.batch = atoi ( argv[n] + 7 ); continue;
		}
	}
	gridProp.act_size_y = 1;

	printf("Grid: %dx1 , %d iterations, %d batches\n", gridProp.logical_size_x, gridProp.num_iter, gridProp.batch);

	//adding halo
	gridProp.act_size_x = gridProp.logical_size_x+2;
	gridProp.act_size_y = 1;

	//padding each row as multiple of vectorization factor
	gridProp.grid_size_x = (gridProp.act_size_x % vectorization_factor) != 0 ?
			(gridProp.act_size_x/vectorization_factor + 1) * vectorization_factor :
			gridProp.act_size_x;
	gridProp.grid_size_y = gridProp.act_size_y;

	//allocating memory buffer
	unsigned int data_size_bytes = gridProp.grid_size_x * gridProp.grid_size_y * sizeof(float) * gridProp.batch;

	if(data_size_bytes >= 4000000000)
	{
		std::cerr << "Maximum buffer size is exceeded!" << std::endl;
		return -1;
	}

	float * grid_u1_cpu = (float*) aligned_alloc(4096, data_size_bytes);
	float * grid_u2_cpu = (float*) aligned_alloc(4096, data_size_bytes);
	float * grid_u3_cpu = (float*) aligned_alloc(4096, data_size_bytes);
	float * grid_u4_cpu = (float*) aligned_alloc(4096, data_size_bytes);

	test_blacksholes_call_option();

	BlacksholesParameter calcParam;

//	calcParam.spot_price = 16;
//	calcParam.strike_price = 10;
//	calcParam.time_to_maturity = 0.25;
//	calcParam.volatility = 0.4;
//	calcParam.risk_free_rate = 0.1;s
	calcParam.spot_price = 16;
	calcParam.strike_price = 10;
	calcParam.time_to_maturity = 0.25;
	calcParam.volatility = 0.4;
	calcParam.risk_free_rate = 0.1;
	calcParam.N = gridProp.num_iter;
	calcParam.K = gridProp.logical_size_x;
	calcParam.SMaxFactor = 3;
	calcParam.delta_t = calcParam.time_to_maturity / calcParam.N;
	calcParam.delta_S = calcParam.strike_price * calcParam.SMaxFactor/ (calcParam.K);

	intialize_grid(grid_u1_cpu, gridProp, calcParam);
	copy_grid(grid_u1_cpu, grid_u2_cpu, gridProp);
	copy_grid(grid_u1_cpu, grid_u3_cpu, gridProp);
	copy_grid(grid_u1_cpu, grid_u4_cpu, gridProp);

	//golden stencil computation on host

	std::cout << "stencil stability: " << stencil_stability(calcParam) << std::endl;

	bs_implicit1(grid_u1_cpu, grid_u2_cpu, gridProp, calcParam);
//	for (unsigned int itr = 0; itr < calcParam.N / 2; itr++)
//	{
//		bs_implicit1(grid_u1_cpu, grid_u2_cpu, gridProp, calcParam);
//		bs_implicit1(grid_u2_cpu, grid_u1_cpu, gridProp, calcParam);
//	}
	bs_implicit_istvan(grid_u3_cpu, grid_u4_cpu, gridProp, calcParam);

	for (unsigned int bat = 0; bat < gridProp.batch; bat++)
	{
//		int offset = bat * gridProp.grid_size_x;
//
//		for (unsigned int i = 0; i < gridProp.act_size_x; i++)
//		{
//
//			std::cout << "grid_id: " << offset  + i << " explicit1 val: " << grid_u1_cpu[offset + i] << " istvan explicit val: " << grid_u3_cpu[offset + i] << std::endl;
//		}

		std::cout << "call option price from explicit method: " << get_call_option(grid_u1_cpu, gridProp, calcParam) << std::endl;
		std::cout << "call option price from istvan explicit method: " << get_call_option(grid_u3_cpu, gridProp, calcParam) << std::endl;
	}

	return 0;
}


