//Implementation to run OPS implementation as standalone with contrast to blacksholes_app.

#include <iostream>
#include "../blacksholes_common.h"
#include "../blacksholes_cpu.h"

#define OPS_1D
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

	//checking stability condition of blacksholes calculation
	if (stencil_stability(calcParam))
	{
		std::cout << "stencil calculation is stable" << std::endl << std::endl;
	}
	else
	{
		std::cerr << "stencil calculation stability check failed" << std::endl << std::endl;
		return -1;
	}

    float * grid_u1_cpu = (float*) aligned_alloc(4096, data_size_bytes);
	float * grid_u2_cpu = (float*) aligned_alloc(4096, data_size_bytes);
	float * grid_ops_result = (float*) aligned_alloc(4096, data_size_bytes);

    //Allocating OPS instance
	OPS_instance * ops_inst = new OPS_instance(argc, argv, 1);

	intialize_grid(grid_u1_cpu, gridProp, calcParam);
	copy_grid(grid_u1_cpu, grid_u2_cpu, gridProp);

    //explicit blacksholes test
    test_blacksholes_call_option(calcParam);

	//golden stencil computation on host
	bs_explicit1(grid_u1_cpu, grid_u2_cpu, gridProp, calcParam);

    // ****************************************************
	// ** ops implementation
	// ****************************************************

		//ops_block
	ops_block grid1D = ops_inst->decl_block(1, "grid1D");

	//ops_data
	int size[] = {static_cast<int>(gridProp.logical_size_x)};
	int base[] = {0};
	int d_m[] = {-1};
	int d_p[] = {1};

	float *current = nullptr, *next = nullptr;
	float *a = nullptr, *b = nullptr, *c = nullptr;

	ops_dat dat_current = grid1D->decl_dat(1, size, base, d_m, d_p, current,"float", "dat_current");
	ops_dat dat_next = grid1D->decl_dat(1, size, base, d_m, d_p, next,"float", "dat_next");
	ops_dat dat_a = grid1D->decl_dat(1, size, base, d_m, d_p, a, "float", "dat_a");
	ops_dat dat_b = grid1D->decl_dat(1, size, base, d_m, d_p, b, "float", "dat_b");
	ops_dat dat_c = grid1D->decl_dat(1, size, base, d_m, d_p, c, "float", "dat_c");
	//ops_constant declarations
//	ops_inst->decl_const("delta_S", 1, "float", calcParam.delta_S);

	//defining the stencil
	int s1d_3pt[] = {-1, 0, 1};
	ops_stencil S1D_3pt = ops_inst->decl_stencil(1, 3, s1d_3pt, "3pt");

	int s1d_1pt[] = {0};
	ops_stencil S1D_1pt = ops_inst->decl_stencil(1, 1, s1d_1pt, "1pt");

	ops_inst->partition("");

	for (int bat = 0; bat < gridProp.batch; bat++)
	{
		int offset 	= bat * gridProp.grid_size_x;
		//initializing data
		int lower_Pad_range[] = {-1,0};
		ops_par_loop(ops_krnl_zero_init, "ops_zero_init", grid1D, 1, lower_Pad_range,
				ops_arg_dat(dat_current, 1, S1D_1pt, "float", OPS_WRITE));

		int upper_pad_range[] = {gridProp.logical_size_x, gridProp.logical_size_x + 1};
		float sMax = calcParam.SMaxFactor * calcParam.strike_price;

		ops_par_loop(ops_krnl_const_init, "ops_const_init", grid1D, 1, upper_pad_range,
				ops_arg_dat(dat_current, 1, S1D_1pt, "float", OPS_WRITE),
				ops_arg_gbl(&sMax, 1, "float", OPS_READ));

		int interior_range[] = {0, gridProp.logical_size_x};
		ops_par_loop(ops_krnl_interior_init, "ïnterior_init", grid1D, 1, interior_range,
				ops_arg_dat(dat_current, 1, S1D_1pt, "float", OPS_WRITE),
				ops_arg_idx(),
				ops_arg_gbl(&(calcParam.delta_S), 1, "float", OPS_READ),
				ops_arg_gbl(&(calcParam.strike_price),1,"float", OPS_READ));

		int full_range[] = {-1, gridProp.logical_size_x + 1};
		ops_par_loop(ops_krnl_copy, "init_dat_next", grid1D, 1, full_range,
				ops_arg_dat(dat_next, 1, S1D_1pt, "float", OPS_WRITE),
				ops_arg_dat(dat_current, 1, S1D_1pt, "float", OPS_READ));

		//blacksholes calc
		float alpha = calcParam.volatility * calcParam.volatility * calcParam.delta_t;
		float beta = calcParam.risk_free_rate * calcParam.delta_t;

		for (int iter = 0 ; iter < calcParam.N; iter+=2)
		{
			ops_par_loop(ops_krnl_blacksholes, "blacksholes_1", grid1D, 1, interior_range,
					ops_arg_dat(dat_current, 1, S1D_1pt, "float", OPS_WRITE),
					ops_arg_dat(dat_next, 1, S1D_3pt, "float", OPS_READ),
					ops_arg_dat(dat_a, 1, S1D_1pt, "float", OPS_RW),
					ops_arg_dat(dat_b, 1, S1D_1pt, "float", OPS_RW),
					ops_arg_dat(dat_c, 1, S1D_1pt, "float", OPS_RW),
					ops_arg_gbl(&(alpha), 1, "float", OPS_READ),
					ops_arg_gbl(&(beta), 1 , "float", OPS_READ),
					ops_arg_idx(),
					ops_arg_gbl(&(iter), 1, "int", OPS_READ));

			int iterPlus1 = iter + 1;

			ops_par_loop(ops_krnl_blacksholes, "blacksholes_2", grid1D, 1, interior_range,
					ops_arg_dat(dat_next, 1, S1D_1pt, "float", OPS_WRITE),
					ops_arg_dat(dat_current, 1, S1D_3pt, "float", OPS_READ),
					ops_arg_dat(dat_a, 1, S1D_1pt, "float", OPS_RW),
					ops_arg_dat(dat_b, 1, S1D_1pt, "float", OPS_RW),
					ops_arg_dat(dat_c, 1, S1D_1pt, "float", OPS_RW),
					ops_arg_gbl(&(alpha), 1, "float", OPS_READ),
					ops_arg_gbl(&(beta), 1 , "float", OPS_READ),
					ops_arg_idx(),
					ops_arg_gbl(&(iterPlus1), 1, "int", OPS_READ));
		}

		//fetching back result
		std::cout << "dat_current_size_x: " << dat_current->size[0] << std::endl;

		std::cout << "local n partition: " << dat_current->get_local_npartitions() << " global n partition: " << dat_current->get_global_npartitions() << std::endl;
		ops_dat_fetch_data_host(dat_current, 0, (char*)(grid_ops_result + offset + 1));
	}



//	for (int i = 0; i < gridProp.logical_size_x; i++)
//	{
//		std::cout << "idx: " << i << ", dat_current: " << grid_ops_result[i] << std::endl;
//	}
//	ops_print_dat_to_txtfile_core(dat_current, "dat_current.txt");
//	ops_print_dat_to_txtfile_core(dat_next, "dat_next.txt");
	

	//Finalizing the OPS library
	delete ops_inst;
    
    free(grid_u1_cpu);
    free(grid_u2_cpu);
    free(grid_ops_result);

    return 0;
}

