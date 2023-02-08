/** @brief main app of the ops implementation of heat3D as standalone 
  * @author Beniel Thileepan
  * 
  */

#include "../heat3D_common.h"
#include "../heat3D_cpu.h"

// #define OPS_3D
#define OPS_1D
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

    //golden stencil computation on the CPU

    auto naive_cpu_start_clk_point = std::chrono::high_resolution_clock::now();
    heat3D_explicit(grid_u1_cpu, grid_u2_cpu, gridData, calcParam);
    auto naive_cpu_stop_clk_point = std::chrono::high_resolution_clock::now();
    double runtime_naive_cpu_stencil = std::chrono::duration<double, std::micro> (naive_cpu_stop_clk_point - naive_cpu_start_clk_point).count();

#ifdef DEBUG_VERBOSE
    std::cout << std::endl;
    std::cout << "*********************************************"  << std::endl;
    std::cout << "**      Debug info after calculations      **"  << std::endl;
    std::cout << "*********************************************"  << std::endl;

    std::cout << "============================================="  << std::endl << std::endl;
#endif

    std::cout << std::endl;
    std::cout << "*********************************************"  << std::endl;
    std::cout << "**            runtime summery              **"  << std::endl;
    std::cout << "*********************************************"  << std::endl;
    std::cout << " * naive stencil runtime   : " << runtime_init + runtime_naive_cpu_stencil << std::endl;
    std::cout << "       |--> grid init time : " << runtime_init << std::endl;
    std::cout << "       |--> calc time      : " << runtime_naive_cpu_stencil << std::endl;
    std::cout << "============================================="  << std::endl << std::endl;

    free(grid_u1_cpu);
    free(grid_u2_cpu);

    return 0;
}

