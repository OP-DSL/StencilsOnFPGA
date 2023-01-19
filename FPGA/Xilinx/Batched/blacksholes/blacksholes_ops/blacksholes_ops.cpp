
#include "blacksholes_ops.h"
#include "blacksholes_ops_kernels.h"

int bs_explicit1_ops(float* result, OPS_instance * ops_inst, GridParameter gridData, BlacksholesParameter computeParam)
{
	//ops_block
	ops_block grid1D = ops_inst->decl_block(1, "grid1D");

	//ops_data
	int size[] = {static_cast<int>(gridData.logical_size_x)};
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
//	ops_inst->decl_const("delta_S", 1, "float", computeParam.delta_S);

	//defining the stencil
	int s1d_3pt[] = {-1, 0, 1};
	ops_stencil S1D_3pt = ops_inst->decl_stencil(1, 3, s1d_3pt, "3pt");

	int s1d_1pt[] = {0};
	ops_stencil S1D_1pt = ops_inst->decl_stencil(1, 1, s1d_1pt, "1pt");

	ops_inst->partition("");

	for (int bat = 0; bat < gridData.batch; bat++)
	{
		int offset 	= bat * gridData.grid_size_x;
		//initializing data
		int lower_Pad_range[] = {-1,0};
		ops_par_loop(ops_krnl_zero_init, "ops_zero_init", grid1D, 1, lower_Pad_range,
				ops_arg_dat(dat_current, 1, S1D_1pt, "float", OPS_WRITE));

		int upper_pad_range[] = {gridData.logical_size_x, gridData.logical_size_x + 1};
		float sMax = computeParam.SMaxFactor * computeParam.strike_price;

		ops_par_loop(ops_krnl_const_init, "ops_const_init", grid1D, 1, upper_pad_range,
				ops_arg_dat(dat_current, 1, S1D_1pt, "float", OPS_WRITE),
				ops_arg_gbl(&sMax, 1, "float", OPS_READ));

		int interior_range[] = {0, gridData.logical_size_x};
		ops_par_loop(ops_krnl_interior_init, "ïnterior_init", grid1D, 1, interior_range,
				ops_arg_dat(dat_current, 1, S1D_1pt, "float", OPS_WRITE),
				ops_arg_idx(),
				ops_arg_gbl(&(computeParam.delta_S), 1, "float", OPS_READ),
				ops_arg_gbl(&(computeParam.strike_price),1,"float", OPS_READ));

		int full_range[] = {-1, gridData.logical_size_x + 1};
		ops_par_loop(ops_krnl_copy, "init_dat_next", grid1D, 1, full_range,
				ops_arg_dat(dat_next, 1, S1D_1pt, "float", OPS_WRITE),
				ops_arg_dat(dat_current, 1, S1D_1pt, "float", OPS_READ));

		//blacksholes calc
		float alpha = computeParam.volatility * computeParam.volatility * computeParam.delta_t;
		float beta = computeParam.risk_free_rate * computeParam.delta_t;

		for (int iter = 0 ; iter < computeParam.N; iter+=2)
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
		ops_dat_fetch_data_host(dat_current, 0, (char*)(result + offset + 1));
	}



//	for (int i = 0; i < gridData.logical_size_x; i++)
//	{
//		std::cout << "idx: " << i << ", dat_current: " << result[i] << std::endl;
//	}
//	ops_print_dat_to_txtfile_core(dat_current, "dat_current.txt");
//	ops_print_dat_to_txtfile_core(dat_next, "dat_next.txt");
	return 0;
}
