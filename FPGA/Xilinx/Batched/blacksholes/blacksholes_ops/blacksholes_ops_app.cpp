
/** @brief Implementation to run OPS implementation as standalone with contrast to blacksholes_app.
  * @author Beniel Thileepan
  * @details
  *
  *  This version is based on C/C++ and uses the OPS prototype highlevel domain
  *  specific API for developing multi-block Structured mesh applications.
  *  Coded in a way both C++ API/ C API can be selected by defining OPS_CPP_API.
  */

#include <iostream>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <chrono>
#include <random>
#include "../blacksholes_common.h"
#include "../blacksholes_cpu.h"

#define OPS_1D
#define OPS_CPP_API
// #define DEBUG_VERBOSE
#include "ops_seq_v2.h"
#include "blacksholes_ops_kernels.h"


int main(int argc, char **argv)
{

	GridParameter gridProp;
	gridProp.logical_size_x = 200;
	gridProp.logical_size_y = 1;
	gridProp.batch = 1;
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

	std::vector<BlacksholesParameter> calcParam(gridProp.batch); //multiple blacksholes calculations

	//First calculation for test value

	calcParam[0].spot_price = 16;
	calcParam[0].strike_price = 10;
	calcParam[0].time_to_maturity = 0.25;
	calcParam[0].volatility = 0.4;
	calcParam[0].risk_free_rate = 0.1;
	calcParam[0].N = gridProp.num_iter;
	calcParam[0].K = gridProp.logical_size_x;
	calcParam[0].SMaxFactor = 3;
	calcParam[0].delta_t = calcParam[0].time_to_maturity / calcParam[0].N;
	calcParam[0].delta_S = calcParam[0].strike_price * calcParam[0].SMaxFactor/ (calcParam[0].K);
	calcParam[0].stable = stencil_stability(calcParam[0]);

	std::random_device dev;
	std::mt19937 rndGen(dev());
	std::uniform_real_distribution<> dis(0.0, 1.0);

	for (int i = 1; i < gridProp.batch; i++)
	{
		calcParam[i].spot_price = 16 + dis(rndGen);
		calcParam[i].strike_price = 10 + dis(rndGen);
		calcParam[i].time_to_maturity = 0.25 + dis(rndGen);
		calcParam[i].volatility = 0.4 + dis(rndGen);
		calcParam[i].risk_free_rate = 0.1 + dis(rndGen);
		calcParam[i].N = gridProp.num_iter;
		calcParam[i].K = gridProp.logical_size_x;
		calcParam[i].SMaxFactor = 3;
		calcParam[i].delta_t = calcParam[i].time_to_maturity / calcParam[i].N;
		calcParam[i].delta_S = calcParam[i].strike_price * calcParam[i].SMaxFactor/ (calcParam[i].K);
		calcParam[i].stable = stencil_stability(calcParam[i]);
	}

    float * grid_u1_cpu = (float*) aligned_alloc(4096, data_size_bytes);
	float * grid_u2_cpu = (float*) aligned_alloc(4096, data_size_bytes);
	float * grid_ops_result = (float*) aligned_alloc(4096, data_size_bytes);

	auto init_start_clk_point = std::chrono::high_resolution_clock::now();
	intialize_grid(grid_u1_cpu, gridProp, calcParam);
	copy_grid(grid_u1_cpu, grid_u2_cpu, gridProp);
	auto init_stop_clk_point = std::chrono::high_resolution_clock::now();
	double runtime_init = std::chrono::duration<double, std::micro> (init_stop_clk_point - init_start_clk_point).count();

#ifdef DEBUG_VERBOSE
	std::cout << std::endl;
	std::cout << "*********************************************"  << std::endl;
	std::cout << "**            intial grid values           **"  << std::endl;
	std::cout << "*********************************************"  << std::endl;

	for (unsigned int bat = 0; bat < gridProp.batch; bat++)
	{
		int offset = bat * gridProp.grid_size_x;

		for (unsigned int i = 0; i < gridProp.act_size_x; i++)
		{
			std::cout << "grid_id: " << offset  + i << " initial_val: " << grid_u1_cpu[offset + i]<< std::endl;
		}
	}
	std::cout << "============================================="  << std::endl << std::endl;
#endif

    //explicit blacksholes test
    double runtime_direct_per_option = test_blacksholes_call_option(calcParam[0]);

	//golden stencil computation on host

	auto naive_start_clk_point = std::chrono::high_resolution_clock::now();
	bs_explicit1(grid_u1_cpu, grid_u2_cpu, gridProp, calcParam);
	auto naive_stop_clk_point = std::chrono::high_resolution_clock::now();
	double runtime_naive_stencil = std::chrono::duration<double, std::micro> (naive_stop_clk_point - naive_start_clk_point).count(); 

    // ****************************************************
	// ** ops implementation
	// ****************************************************
	
	auto ops_init_start_clk_point = std::chrono::high_resolution_clock::now();

#ifdef OPS_CPP_API
    //Allocating OPS instance
	OPS_instance * ops_inst = new OPS_instance(argc, argv, 1);
#else
	//OPS initialization 
	ops_init(argc, argv, 1);
#endif

	//ops_block
#ifdef OPS_CPP_API
	ops_block grid1D = ops_inst->decl_block(1, "grid1D");
#else
	ops_block grid1D = ops_decl_block(1, "grid1D");
#endif

	//ops_data
	int size[] = {static_cast<int>(gridProp.logical_size_x)};
	int base[] = {0};
	int d_m[] = {-1};
	int d_p[] = {1};

	float *current = nullptr, *next = nullptr;
	float *a = nullptr, *b = nullptr, *c = nullptr;

#ifdef OPS_CPP_API
	ops_dat dat_current = grid1D->decl_dat(1, size, base, d_m, d_p, current,"float", "dat_current");
	ops_dat dat_next = grid1D->decl_dat(1, size, base, d_m, d_p, next,"float", "dat_next");
	ops_dat dat_a = grid1D->decl_dat(1, size, base, d_m, d_p, a, "float", "dat_a");
	ops_dat dat_b = grid1D->decl_dat(1, size, base, d_m, d_p, b, "float", "dat_b");
	ops_dat dat_c = grid1D->decl_dat(1, size, base, d_m, d_p, c, "float", "dat_c");
#else
	ops_dat dat_current = ops_decl_dat(grid1D, 1, size, base, d_m, d_p, current,"float", "dat_current");
	ops_dat dat_next = ops_decl_dat(grid1D, 1, size, base, d_m, d_p, next,"float", "dat_next");
	ops_dat dat_a = ops_decl_dat(grid1D, 1, size, base, d_m, d_p, a, "float", "dat_a");
	ops_dat dat_b = ops_decl_dat(grid1D, 1, size, base, d_m, d_p, b, "float", "dat_b");
	ops_dat dat_c = ops_decl_dat(grid1D, 1, size, base, d_m, d_p, c, "float", "dat_c");
#endif

	//defining the stencils
	int s1d_3pt[] = {-1, 0, 1};
	int s1d_1pt[] = {0};

#ifdef OPS_CPP_API
	ops_stencil S1D_3pt = ops_inst->decl_stencil(1, 3, s1d_3pt, "3pt");
	ops_stencil S1D_1pt = ops_inst->decl_stencil(1, 1, s1d_1pt, "1pt");
#else
	ops_stencil S1D_3pt = ops_decl_stencil(1, 3, s1d_3pt, "3pt");
	ops_stencil S1D_1pt = ops_decl_stencil(1, 1, s1d_1pt, "1pt");
#endif

	//partition
#ifdef OPS_CPP_API
	ops_inst->partition("1D_BLOCK_DECOMPOSE");
#else
	ops_partition("1D_BLOCK_DECOMPOSE");
#endif

	double runtime_blacksholes_kernel = 0.0;
	double runtime_grid_init_kernels = 0.0;
	double runtime_calc_coefficient = 0.0;
	double runtime_device_to_host = 0.0;

	auto ops_init_stop_clk_point = std::chrono::high_resolution_clock::now();
	double runtime_ops_init = std::chrono::duration<double, std::micro> (ops_init_stop_clk_point - ops_init_start_clk_point).count();
	auto ops_start_clk_point = ops_init_stop_clk_point;

	for (int bat = 0; bat < gridProp.batch; bat++)
	{
		auto grid_init_start_clk_point = std::chrono::high_resolution_clock::now();

		int offset 	= bat * gridProp.grid_size_x;
		//initializing data
		int lower_Pad_range[] = {-1,0};
		ops_par_loop(ops_krnl_zero_init, "ops_zero_init", grid1D, 1, lower_Pad_range,
				ops_arg_dat(dat_current, 1, S1D_1pt, "float", OPS_WRITE));

		int upper_pad_range[] = {gridProp.logical_size_x, gridProp.logical_size_x + 1};
		float sMax = calcParam[bat].SMaxFactor * calcParam[bat].strike_price;

		ops_par_loop(ops_krnl_const_init, "ops_const_init", grid1D, 1, upper_pad_range,
				ops_arg_dat(dat_current, 1, S1D_1pt, "float", OPS_WRITE),
				ops_arg_gbl(&sMax, 1, "float", OPS_READ));

		int interior_range[] = {0, gridProp.logical_size_x};
		ops_par_loop(ops_krnl_interior_init, "interior_init", grid1D, 1, interior_range,
				ops_arg_dat(dat_current, 1, S1D_1pt, "float", OPS_WRITE),
				ops_arg_idx(),
				ops_arg_gbl(&(calcParam[bat].delta_S), 1, "float", OPS_READ),
				ops_arg_gbl(&(calcParam[bat].strike_price),1,"float", OPS_READ));

		int full_range[] = {-1, gridProp.logical_size_x + 1};
		ops_par_loop(ops_krnl_copy, "init_dat_next", grid1D, 1, full_range,
				ops_arg_dat(dat_next, 1, S1D_1pt, "float", OPS_WRITE),
				ops_arg_dat(dat_current, 1, S1D_1pt, "float", OPS_READ));

		auto grid_init_stop_clk_point = std::chrono::high_resolution_clock::now();

		runtime_grid_init_kernels += std::chrono::duration<double, std::micro>(grid_init_stop_clk_point - grid_init_start_clk_point).count();

		auto coeff_calc_start_clk_point = std::chrono::high_resolution_clock::now();
	
		//blacksholes calc
		float alpha = calcParam[bat].volatility * calcParam[bat].volatility * calcParam[bat].delta_t;
		float beta = calcParam[bat].risk_free_rate * calcParam[bat].delta_t;

		ops_par_loop(ops_krnl_calc_coefficient, "calc_coefficient", grid1D, 1, interior_range,
				ops_arg_dat(dat_a, 1, S1D_1pt, "float", OPS_WRITE),
				ops_arg_dat(dat_b, 1, S1D_1pt, "float", OPS_WRITE),
				ops_arg_dat(dat_c, 1, S1D_1pt, "float", OPS_WRITE),
				ops_arg_gbl(&(alpha), 1, "float", OPS_READ),
				ops_arg_gbl(&(beta), 1 , "float", OPS_READ),
				ops_arg_idx());

		auto coeff_calc_stop_clk_point = std::chrono::high_resolution_clock::now();
		
		runtime_calc_coefficient += std::chrono::duration<double, std::micro>(coeff_calc_stop_clk_point - coeff_calc_start_clk_point).count();

		auto blacksholes_calc_start_clk_point = coeff_calc_stop_clk_point;

		for (int iter = 0 ; iter < calcParam[bat].N; iter+=2)
		{
			ops_par_loop(ops_krnl_blacksholes, "blacksholes_1", grid1D, 1, interior_range,
					ops_arg_dat(dat_current, 1, S1D_1pt, "float", OPS_WRITE),
					ops_arg_dat(dat_next, 1, S1D_3pt, "float", OPS_READ),
					ops_arg_dat(dat_a, 1, S1D_1pt, "float", OPS_READ),
					ops_arg_dat(dat_b, 1, S1D_1pt, "float", OPS_READ),
					ops_arg_dat(dat_c, 1, S1D_1pt, "float", OPS_READ));


			ops_par_loop(ops_krnl_blacksholes, "blacksholes_2", grid1D, 1, interior_range,
					ops_arg_dat(dat_next, 1, S1D_1pt, "float", OPS_WRITE),
					ops_arg_dat(dat_current, 1, S1D_3pt, "float", OPS_READ),
					ops_arg_dat(dat_a, 1, S1D_1pt, "float", OPS_READ),
					ops_arg_dat(dat_b, 1, S1D_1pt, "float", OPS_READ),
					ops_arg_dat(dat_c, 1, S1D_1pt, "float", OPS_READ));
		}

		auto blacksholes_calc_stop_clk_point = std::chrono::high_resolution_clock::now();

		runtime_blacksholes_kernel += std::chrono::duration<double, std::micro>(blacksholes_calc_stop_clk_point - blacksholes_calc_start_clk_point).count();

#ifdef DEBUG_VERBOSE
		//fetching back result
		ops_dat_fetch_data_host(dat_current, 0, (char*)(grid_ops_result + offset + 1));
		auto device_to_host_stop_clk_point = std::chrono::high_resolution_clock::now();
		runtime_device_to_host += std::chrono::duration<double, std::micro>(device_to_host_stop_clk_point - blacksholes_calc_stop_clk_point).count();
#endif
	}

	auto ops_stop_clk_point = std::chrono::high_resolution_clock::now();
	double runtime_ops_stencil = std::chrono::duration<double, std::micro> (ops_stop_clk_point - ops_start_clk_point).count();

	std::cout << std::endl;
	std::cout << "*********************************************"  << std::endl;
	std::cout << "**      Debug info after calculations      **"  << std::endl;
	std::cout << "*********************************************"  << std::endl;

#ifdef DEBUG_VERBOSE
	for (unsigned int bat = 0; bat < gridProp.batch; bat++)
	{
		int offset = bat * gridProp.grid_size_x;

		for (unsigned int i = 0; i < gridProp.act_size_x; i++)
		{

			std::cout << "grid_id: " << offset  + i << " explicit1_val: " << grid_u1_cpu[offset + i]
						<< " explicit1_ops_val: " << grid_ops_result[offset + i]<< std::endl;
		}
	}

	std::cout << "call option[0] price from explicit method: " << get_call_option(grid_u1_cpu, calcParam[0]) << std::endl;
	std::cout << "call option[0] price from ops explicit method: " << get_call_option(grid_ops_result, calcParam[0]) << std::endl;

	std::cout << "============================================="  << std::endl << std::endl;
#endif
	
//	for (int i = 0; i < gridProp.logical_size_x; i++)
//	{
//		std::cout << "idx: " << i << ", dat_current: " << grid_ops_result[i] << std::endl;
//	}
//	ops_print_dat_to_txtfile_core(dat_current, "dat_current.txt");
//	ops_print_dat_to_txtfile_core(dat_next, "dat_next.txt");
	
	std::cout << std::endl;
	std::cout << "*********************************************"  << std::endl;
	std::cout << "**            runtime summery              **"  << std::endl;
	std::cout << "*********************************************"  << std::endl;

	std::cout << " * direct_cal time         : " << runtime_direct_per_option * gridProp.batch << "us" << std::endl;
	std::cout << "      |--> time_per_option : " << runtime_direct_per_option << "us" << std::endl;
	std::cout << " * naive stencil runtime   : " << runtime_init + runtime_naive_stencil << "us" << std::endl;
	std::cout << "      |--> grid_init time  : " << runtime_init << "us" << std::endl;
	std::cout << "      |--> calc time       : " << runtime_naive_stencil << "us" << std::endl;
	std::cout << " * ops resource alloc time : " << runtime_ops_init << "us" << std::endl;
	std::cout << " * ops stencil runtime     : " << runtime_ops_stencil<< "us" << std::endl;
	std::cout << "      |--> grid_init time  : " << runtime_grid_init_kernels << "us" << std::endl;
	std::cout << "      |--> calc coef. time : " << runtime_calc_coefficient << "us" << std::endl;
	std::cout << "      |--> DtoH time       : " << runtime_device_to_host << "us" << std::endl;
	std::cout << "      |--> calc time       : " << runtime_blacksholes_kernel << "us" << std::endl;
	std::cout << "============================================="  << std::endl << std::endl;
	//Finalizing the OPS library
#ifdef OPS_CPP_API
	delete ops_inst;
#else
	ops_exit();
#endif

    free(grid_u1_cpu);
    free(grid_u2_cpu);
    free(grid_ops_result);

    return 0;
}

