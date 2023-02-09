/** @brief main app of the ops implementation of heat3D as standalone 
  * @author Beniel Thileepan
  * 
  */

#include "../heat3D_common.h"
#include "../heat3D_cpu.h"

#define OPS_3D
#define OPS_CPP_API
//#define DEBUG_VERBOSE
#include "ops_seq_v2.h"
#include "heat3D_ops_kernels.h"

int main(int argc, char **argv)
{
    GridParameter gridData;
    gridData.logical_size_x = 200;
    gridData.logical_size_y = 200;
    gridData.logical_size_z = 200;
    gridData.batch = 10;
    gridData.num_iter = 100;

    unsigned int vectorization_factor = 8;
    
    // setting grid parameters given by user
    const char * pch;

    for ( int n = 1; n < argc; n++ )
    {
        pch = strstr(argv[n], "-size=");

        if(pch != NULL)
        {
            gridData.logical_size_x = atoi ( argv[n] + 7 ); continue;
        }

        pch = strstr(argv[n], "-iters=");

        if(pch != NULL)
        {
            gridData.num_iter = atoi ( argv[n] + 7 ); continue;
        }
        pch = strstr(argv[n], "-batch=");

        if(pch != NULL)
        {
            gridData.batch = atoi ( argv[n] + 7 ); continue;
        }     
    }

    printf("Grid: %dx1 , %d iterations, %d batches\n", gridData.logical_size_x, gridData.num_iter, gridData.batch);

    //adding halo
    gridData.act_size_x = gridData.logical_size_x + 2;
    gridData.act_size_y = gridData.logical_size_y + 2;
    gridData.act_size_z = gridData.logical_size_z + 2;

    //padding each row as multiples of vectorization factor
    gridData.grid_size_x = (gridData.act_size_x % vectorization_factor) != 0 ?
			      (gridData.act_size_x/vectorization_factor + 1) * vectorization_factor :
			      gridData.act_size_x;
	  gridData.grid_size_y = gridData.act_size_y;
    gridData.grid_size_z = gridData.act_size_z;

    //allocating memory buffer
    unsigned int data_size_bytes = gridData.grid_size_x * gridData.grid_size_y 
            * gridData.grid_size_z * sizeof(float) * gridData.batch;
    
    if (data_size_bytes >= 4000000000)
    {
        std::cerr << "Maximum buffer size is exceeded!" << std::endl;
    }

    std::vector<heat3DParameter> calcParam(gridData.batch);

    for (unsigned int bat = 0; bat < gridData.batch; bat++)
    {

        calcParam[bat].alpha = 1.5/1000; //diffusivity 
        calcParam[bat].h = 1/gridData.act_size_x; 
        calcParam[bat].delta_t = 0.5; //0.5s
        calcParam[bat].K = calcParam[bat].alpha * calcParam[bat].delta_t / (calcParam[bat].h * calcParam[bat].h);
    }

    // ****************************************************
	// ** golden stencil computation on the CPU
	// ****************************************************

    float * grid_u1_cpu = (float*) aligned_alloc(4096, data_size_bytes);
    float * grid_u2_cpu = (float*) aligned_alloc(4096, data_size_bytes);

    auto init_start_clk_point = std::chrono::high_resolution_clock::now();
    initialize_grid(grid_u1_cpu, gridData);
    copy_grid(grid_u1_cpu, grid_u2_cpu, gridData);
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

    auto naive_cpu_start_clk_point = std::chrono::high_resolution_clock::now();
    heat3D_explicit(grid_u1_cpu, grid_u2_cpu, gridData, calcParam);
    auto naive_cpu_stop_clk_point = std::chrono::high_resolution_clock::now();
    double runtime_naive_cpu_stencil = std::chrono::duration<double, std::micro> (naive_cpu_stop_clk_point - naive_cpu_start_clk_point).count();


    // ****************************************************
	// ** ops implementation
	// ****************************************************

    auto ops_init_start_clk_point = std::chrono::high_resolution_clock::now();

    OPS_instance * ops_inst = new OPS_instance(argc, argv, 1);

    ops_block grid = ops_inst->decl_block(3, "grid");

    int size[] = {gridData.logical_size_x, gridData.logical_size_y, gridData.logical_size_z};
    int base[] = {0, 0, 0};
    int d_m[] = {-1, -1, -1};
    int d_p[] = {1, 1, 1};

    float * current = nullptr, * next = nullptr;
    float * grid_ops_result = (float*) aligned_alloc(4096, data_size_bytes);

    ops_dat dat_current = grid->decl_dat(3, size, base, d_m, d_p, current,  "float", "dat_current");
    ops_dat dat_next = grid->decl_dat(3, size, base, d_m, d_p, next, "float", "dat_next");

    //defining stencils
    int s3d_1pt[] = {0,0,0};
    ops_stencil stencil3D_1pt = ops_inst->decl_stencil(3, 1, s3d_1pt, "1pt stencil");

    int s3d_7pt[] = {0,0,0, 1,0,0, -1,0,0, 0,1,0, 0,-1,0, 0,0,1, 0,0,-1};
    ops_stencil stencil3D_7pt = ops_inst->decl_stencil(3, 7, s3d_7pt, "7pt stencil");

    //Reduction handle
    ops_reduction h_err = ops_inst->decl_reduction_handle(sizeof(float), "float", "error");

    //paritioning (not functioning yet in OPS)
    ops_inst->partition("");

    double runtime_heat3D_kernel = 0.0;
    double runtime_grid_init_kernels = 0.0;
#ifdef DEBUG_VERBOSE
    double runtime_device_to_host = 0.0;
#endif

    auto ops_init_stop_clk_point = std::chrono::high_resolution_clock::now();
    double runtime_ops_init = std::chrono::duration<double, std::micro>(ops_init_stop_clk_point - ops_init_start_clk_point).count();

    auto ops_start_clk_point = std::chrono::high_resolution_clock::now();

    //defining the access ranges
    int bottom_plane_range[] = {-1,-1,-1,  gridData.logical_size_x+1,gridData.logical_size_y+1, -1};
    int top_plane_range[] = {-1,-1,gridData.logical_size_z+1,  gridData.logical_size_x+1,gridData.logical_size_y+1,gridData.logical_size_z+1};
    int front_plane_range[] = {-1,-1,-1,  gridData.logical_size_x+1,-1,gridData.logical_size_z+1};
    int back_plane_range[] = {-1,gridData.logical_size_y+1,-1,  gridData.logical_size_x+1,gridData.logical_size_y+1,gridData.logical_size_z+1};
    int left_plane_range[] = {-1,-1,-1,  -1,gridData.logical_size_y+1,gridData.logical_size_z+1};
    int right_plane_range[] = {gridData.logical_size_x+1,-1,-1,  gridData.logical_size_x+1,gridData.logical_size_y+1,gridData.logical_size_z+1};
    int internal_range[] = {-1,-1,-1,  gridData.logical_size_x+1,gridData.logical_size_y+1,gridData.logical_size_z+1};

    for (unsigned int bat = 0; bat < gridData.batch; bat++)
    {
        auto grid_init_start_clk_point = std::chrono::high_resolution_clock::now();

        int offeset = bat * gridData.grid_size_x * gridData.grid_size_y * gridData.grid_size_y;

        //Initializing data
        ops_par_loop(ops_krnl_zero_init, "ops_top_plane_init", grid, 3, top_plane_range,
                ops_arg_dat(dat_current, 3, stencil3D_1pt, "float", OPS_WRITE));

        ops_par_loop(ops_krnl_zero_init, "ops_bottom_plane_init", grid, 3, bottom_plane_range,
                ops_arg_dat(dat_current, 3, stencil3D_1pt, "float", OPS_WRITE));

        ops_par_loop(ops_krnl_zero_init, "ops_front_plane_init", grid, 3, front_plane_range,
                ops_arg_dat(dat_current, 3, stencil3D_1pt, "float", OPS_WRITE));
        
        ops_par_loop(ops_krnl_zero_init, "ops_back_plane_init", grid, 3, back_plane_range,
                ops_arg_dat(dat_current, 3, stencil3D_1pt, "float", OPS_WRITE));

        ops_par_loop(ops_krnl_zero_init, "ops_left_plane_init", grid, 3, left_plane_range,
                ops_arg_dat(dat_current, 3, stencil3D_1pt, "float", OPS_WRITE));

        ops_par_loop(ops_krnl_zero_init, "ops_right_plane_init", grid, 3, right_plane_range,
                ops_arg_dat(dat_current, 3, stencil3D_1pt, "float", OPS_WRITE));
        
        auto grid_init_stop_clk_point = std::chrono::high_resolution_clock::now();
        runtime_grid_init_kernels += std::chrono::duration<double, std::micro>(grid_init_stop_clk_point - grid_init_start_clk_point).count();

        auto heat3D_calc_start_clk_point = std::chrono::high_resolution_clock::now();
        unsigned int iter = 0;
        float error = 1;

        while (error > ERROR_TOL and iter < gridData.num_iter)
        {
            ops_par_loop(ops_krnl_heat3D, "ops_krnl_heat3D_1", grid, 3, internal_range,
                    ops_arg_dat(dat_next, 3, stencil3D_1pt, "float", OPS_WRITE),
                    ops_arg_dat(dat_current, 3, stencil3D_7pt, "float", OPS_READ),
                    ops_arg_gbl(&(calcParam[bat].K), 1, "float", OPS_READ),
                    ops_arg_idx(),
                    ops_arg_reduce(h_err, 1, "float", OPS_MAX));
            
            ops_par_loop(ops_krnl_heat3D, "ops_krnl_heat3D_1", grid, 3, internal_range,
                    ops_arg_dat(dat_current, 3, stencil3D_1pt, "float", OPS_WRITE),
                    ops_arg_dat(dat_next, 3, stencil3D_7pt, "float", OPS_READ),
                    ops_arg_gbl(&(calcParam[bat].K), 1, "float", OPS_READ),
                    ops_arg_idx(),
                    ops_arg_reduce(h_err, 1, "float", OPS_MAX));

            // ops_reduction_result(h_err, &error);

#ifdef DEBUG_VERBOSE
            if (iter % 10 == 0) ops_print("iter: %5d, error: %0.6f\n", iter, error);
#endif
            iter += 2;
        }

        auto heat3D_calc_stop_clk_point = std::chrono::high_resolution_clock::now();
        runtime_heat3D_kernel += std::chrono::duration<double, std::micro>(heat3D_calc_stop_clk_point - heat3D_calc_start_clk_point).count();

        if (iter < gridData.num_iter)
        {
            ops_printf("error is less than the tolarance. exiting loop @ iter: %5d, error: %0.6f\n", iter, error);
        }
#ifdef DEBUG_VERBOSE
        auto device_to_host_start_clk_point = std::chrono::high_resolution_clock::now();
        //fetching back result
        ops_dat_fetch_data_host(data_current, 0, (char*)(grid_ops_result + gridData.grid_size_x * gridData.grid_size_y + gridData.grid_size_x + 1));
        auto device_to_host_stop_clk_point = std::chrono::high_resolution_clock::now();

        runtime_device_to_host += std::chrono::duration<double, std::micro>(device_to_host_stop_clk_point - device_top_host_clk_start_point).count();
#endif
    }

    auto ops_end_clk_point = std::chrono::high_resolution_clock::now();
    double runtime_ops_stencil = std::chrono::duration<double, std::micro>(ops_end_clk_point - ops_start_clk_point).count();

#ifdef DEBUG_VERBOSE
    std::cout << std::endl;
    std::cout << "*********************************************"  << std::endl;
    std::cout << "**      Debug info after calculations      **"  << std::endl;
    std::cout << "*********************************************"  << std::endl;
                 
    for (unsigned int bat = 0; bat < gridData.batch; bat++)
    {
        int offeset = bat * gridData.grid_size_x * gridData.grid_size_y * gridData.grid_size_y;

        std::cout << "---------------------------------------------" << std::endl;
        std::cout << "               batch: " << bat << std::endl;
        std::cout << "---------------------------------------------" << std::endl;
        for (unsigned int k = 0; k < gridData.grid_size_z; k++)
        {
            for (unsigned int j = 0; j < gridData.grid_size_y; j++)
            {
                for (unsigned int i = 0; i < gridData.grid_size_x; i++)
                {
                    int index = offset + k * gridData.grid_size_x * gridData.grid_size_y 
                            + j * gridData.grid_size_x + i;
                    std::cout << "grid_id: (" << i << ", " << j << ", " << k << "), golden_val: " << grid_u1_cpu[index]
                            << ", ops_val: " << grid_ops_result[index] << std::endl; 
                }
            }
        }
    }

    std::cout << "============================================="  << std::endl << std::endl;
#endif

    std::cout << std::endl;
    std::cout << "*********************************************"  << std::endl;
    std::cout << "**            runtime summery              **"  << std::endl;
    std::cout << "*********************************************"  << std::endl;
    std::cout << " * naive stencil runtime       : " << runtime_init + runtime_naive_cpu_stencil << std::endl;
    std::cout << "       |--> grid init time     : " << runtime_init << std::endl;
    std::cout << "       |--> calc time          : " << runtime_naive_cpu_stencil << std::endl;
    std::cout << " * ops stencil runtime         : " << runtime_ops_stencil << std::endl;
    std::cout << "       |--> grid init time     : " << runtime_grid_init_kernels << std::endl;
    std::cout << "       |--> calc time          : " << runtime_heat3D_kernel << std::endl;
#ifdef DEBUG_VERBOSE
    std::cout << "       |--> device to host time: " << runtime_init << std::endl;
#endif
    std::cout << "============================================="  << std::endl << std::endl;

    free(grid_u1_cpu);
    free(grid_u2_cpu);
    free(grid_ops_result);

    return 0;
}

