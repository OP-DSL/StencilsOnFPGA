
// standard headers
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <chrono>
#include <iostream>
#include "xcl2.hpp"
#include "heat3D_cpu.h"
#include "heat3D_common.h"

//#define DEBUG_VERBOSE
#define VERIFICATION
#define MULTI_SLR
//#define FPGA_RUN_ONLY

int main(int argc, char **argv)
{
    GridParameter gridData;

    gridData.logical_size_x = 100;
    gridData.logical_size_y = 100;
    gridData.logical_size_z = 100;
    gridData.batch = 10;
    gridData.num_iter = 1000;

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

    heat3DParameter calcParam;


	calcParam.alpha = 1.5/1000; //diffusivity
	calcParam.h = 1/gridData.act_size_x;
	calcParam.delta_t = 0.5; //0.5s
	calcParam.K = calcParam.alpha * calcParam.delta_t / (calcParam.h * calcParam.h);

	float * grid_u1_cpu = (float*) aligned_alloc(4096, data_size_bytes);
	float * grid_u2_cpu = (float*) aligned_alloc(4096, data_size_bytes);

	float * grid_u1_d = (float*) aligned_alloc(4096, data_size_bytes);
	float * grid_u2_d = (float*) aligned_alloc(4096, data_size_bytes);


    auto init_start_clk_point = std::chrono::high_resolution_clock::now();
    initialize_grid(grid_u1_cpu, gridData);
    copy_grid(grid_u1_cpu, grid_u1_d, gridData);
    auto init_stop_clk_point = std::chrono::high_resolution_clock::now();
    double runtime_init = std::chrono::duration<double, std::micro> (init_stop_clk_point - init_start_clk_point).count();
    copy_grid(grid_u1_cpu, grid_u2_cpu, gridData);


#ifdef DEBUG_VERBOSE
    std::cout << std::endl;
    std::cout << "*********************************************"  << std::endl;
    std::cout << "**            intial grid values           **"  << std::endl;
    std::cout << "*********************************************"  << std::endl;

    for (unsigned int bat = 0; bat < gridData.batch; bat++)
    {
        int offset = bat * gridData.grid_size_x * gridData.grid_size_y * gridData.grid_size_y;

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
                	std::cout << "grid_id: (" << i << ", " << j << ", " << k << ") initial_val: "
                			<< grid_u1_cpu[index]<< std::endl;
                }
            }
        }
    }
    std::cout << "============================================="  << std::endl << std::endl;
#endif

#ifndef FPGA_RUN_ONLY
    //golden stencil computation on the CPU

    std::vector<heat3DParameter> calcParams(gridData.batch);

    for (unsigned int bat = 0; bat < gridData.batch; bat++)
    {
    	calcParams[bat] = calcParam;
    }

    auto naive_cpu_start_clk_point = std::chrono::high_resolution_clock::now();
    heat3D_explicit(grid_u1_cpu, grid_u2_cpu, gridData, calcParams);
    auto naive_cpu_stop_clk_point = std::chrono::high_resolution_clock::now();
    double runtime_naive_cpu_stencil = std::chrono::duration<double, std::micro> (naive_cpu_stop_clk_point - naive_cpu_start_clk_point).count();

#endif
    //OPENCL HOST CODE START
    auto bindaryFile = argv[1];
    cl_int err;

    auto devices = xcl::get_xil_devices();
    auto device = devices[0];

    OCL_CHECK(err, cl::Context context(device, NULL, NULL, NULL, &err));
    OCL_CHECK(err, cl::CommandQueue queue(context, CL_QUEUE_PROFILING_ENABLE | CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, &err));
    OCL_CHECK(err, std::string device_name = device.getInfo<CL_DEVICE_NAME>(&err));

    //Create Program and Kernel
    auto fileBuf = xcl::read_binary_file(bindaryFile);
    cl::Program::Binaries bins{{fileBuf.data(), fileBuf.size()}};

    OCL_CHECK(err, cl::Program program(context, {device}, bins, NULL, &err));
    OCL_CHECK(err, cl::Kernel krnl_slr0(program, "stencil_SLR", &err));

#ifdef MULTI_SLR
    OCL_CHECK(err, cl::Kernel krnl_slr1(program, "stencil_SLR", &err));
    OCL_CHECK(err, cl::Kernel krnl_slr2(program, "stencil_SLR", &err));
#endif
    OCL_CHECK(err, cl::Kernel krnl_mem2stream(program, "stencil_mem2stream", &err));

    //Allocation Buffer in Global Memory
    OCL_CHECK(err, cl::Buffer buff_curr(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, data_size_bytes, grid_u1_d, &err));
    OCL_CHECK(err, cl::Buffer buff_next(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, data_size_bytes, grid_u2_d, &err));

#ifdef MULTI_SLR
    unsigned int total_SLR = 3;
#else
    unsigned int total_SLR = 1;
#endif

    unsigned number_of_process_grid_per_SLR = NUM_OF_PROCESS_GRID_PER_SLR;
    unsigned int total_process_grid_per_iter = total_SLR * number_of_process_grid_per_SLR * 2;
    unsigned int num_iter = gridData.num_iter / total_process_grid_per_iter;

    //set Kernel arguments

    /*
     * 	void stencil_SLR(
			const int sizex,
			const int sizey,
			const int sizez,
			const int xdim0,
			const int batches,
			const int count,
			const float calcParam_K,
			hls::stream <t_pkt> &in,
			hls::stream <t_pkt> &out)
     */
    int narg = 0;
    OCL_CHECK(err, err = krnl_slr0.setArg(narg++, gridData.logical_size_x));
    OCL_CHECK(err, err = krnl_slr0.setArg(narg++, gridData.logical_size_y));
    OCL_CHECK(err, err = krnl_slr0.setArg(narg++, gridData.logical_size_z));
    OCL_CHECK(err, err = krnl_slr0.setArg(narg++, gridData.grid_size_x));
    OCL_CHECK(err, err = krnl_slr0.setArg(narg++, gridData.batch));
    OCL_CHECK(err, err = krnl_slr0.setArg(narg++, num_iter));
    OCL_CHECK(err, err = krnl_slr0.setArg(narg++, calcParam.K));

#ifdef MULTI_SLR
    narg = 0;
    OCL_CHECK(err, err = krnl_slr1.setArg(narg++, gridData.logical_size_x));
    OCL_CHECK(err, err = krnl_slr1.setArg(narg++, gridData.logical_size_y));
    OCL_CHECK(err, err = krnl_slr1.setArg(narg++, gridData.logical_size_z));
    OCL_CHECK(err, err = krnl_slr1.setArg(narg++, gridData.grid_size_x));
    OCL_CHECK(err, err = krnl_slr1.setArg(narg++, gridData.batch));
    OCL_CHECK(err, err = krnl_slr1.setArg(narg++, num_iter));
    OCL_CHECK(err, err = krnl_slr1.setArg(narg++, calcParam.K));

    narg = 0;
    OCL_CHECK(err, err = krnl_slr2.setArg(narg++, gridData.logical_size_x));
    OCL_CHECK(err, err = krnl_slr2.setArg(narg++, gridData.logical_size_y));
    OCL_CHECK(err, err = krnl_slr2.setArg(narg++, gridData.logical_size_z));
    OCL_CHECK(err, err = krnl_slr2.setArg(narg++, gridData.grid_size_x));
    OCL_CHECK(err, err = krnl_slr2.setArg(narg++, gridData.batch));
    OCL_CHECK(err, err = krnl_slr2.setArg(narg++, num_iter));
    OCL_CHECK(err, err = krnl_slr2.setArg(narg++, calcParam.K));
#endif

    /*
     * 	void stencil_mem2stream(
			uint512_dt* arg0,
			uint512_dt* arg1,
			const int count,
			const int xdim0,
			const int ydim0,
			const int zdim0,
			const int batch,
			hls::stream <t_pkt> &in,
			hls::stream <t_pkt> &out)
     */
    narg = 0;
    OCL_CHECK(err, err = krnl_mem2stream.setArg(narg++, buff_curr));
    OCL_CHECK(err, err = krnl_mem2stream.setArg(narg++, buff_next));
    OCL_CHECK(err, err = krnl_mem2stream.setArg(narg++, num_iter));
    OCL_CHECK(err, err = krnl_mem2stream.setArg(narg++, gridData.grid_size_x));
    OCL_CHECK(err, err = krnl_mem2stream.setArg(narg++, gridData.grid_size_y));
    OCL_CHECK(err, err = krnl_mem2stream.setArg(narg++, gridData.grid_size_z));
    OCL_CHECK(err, err = krnl_mem2stream.setArg(narg++, gridData.batch));

    //Copy input buffer to device
    auto h_to_d_start_point = std::chrono::high_resolution_clock::now();
    OCL_CHECK(err, err = queue.enqueueMigrateMemObjects({buff_curr, buff_next}, 0));
    queue.finish();
    auto h_to_d_stop_kernels_start_point = std::chrono::high_resolution_clock::now();
#ifdef MULTI_SLR
    OCL_CHECK(err, err = queue.enqueueTask(krnl_slr2));
    OCL_CHECK(err, err = queue.enqueueTask(krnl_slr1));
#endif
    OCL_CHECK(err, err = queue.enqueueTask(krnl_slr0));
    OCL_CHECK(err, err = queue.enqueueTask(krnl_mem2stream));
    queue.finish();
    auto kernels_stop_d_to_h_start_point = std::chrono::high_resolution_clock::now();
    OCL_CHECK(err, err = queue.enqueueMigrateMemObjects({buff_curr}, CL_MIGRATE_MEM_OBJECT_HOST));
    queue.finish();
    auto d_to_h_stop_point = std::chrono::high_resolution_clock::now();

    double h_to_d_runtime = std::chrono::duration<double, std::micro>
    		(h_to_d_stop_kernels_start_point - h_to_d_start_point).count();
    double kernels_runtime = std::chrono::duration<double, std::micro>
    		(kernels_stop_d_to_h_start_point - h_to_d_stop_kernels_start_point).count();
    double d_to_h_runtime = std::chrono::duration<double, std::micro>
    		(d_to_h_stop_point - kernels_stop_d_to_h_start_point).count();

#ifdef VERIFICATION
    std::cout << std::endl;
    std::cout << "*********************************************"  << std::endl;
    std::cout << "**               Verification              **"  << std::endl;
    std::cout << "*********************************************"  << std::endl;

    for (unsigned int bat = 0; bat < gridData.batch; bat++)
    {
        int offset = bat * gridData.grid_size_x * gridData.grid_size_y * gridData.grid_size_y;

        std::cout << "---------------------------------------------" << std::endl;
        std::cout << "               batch: " << bat << std::endl;
        std::cout << "---------------------------------------------" << std::endl;

        bool passed = true;

        for (unsigned int k = 0; k < gridData.grid_size_z; k++)
        {
            for (unsigned int j = 0; j < gridData.grid_size_y; j++)
            {
                for (unsigned int i = 0; i < gridData.grid_size_x; i++)
                {
                	int index = offset + k * gridData.grid_size_x * gridData.grid_size_y
                	                                + j * gridData.grid_size_x + i;
                	if (abs(grid_u1_cpu[index] - grid_u1_d[index]) > EPSILON)
                	{
                		std::cerr << "Value Mismatch index: (" << i << ", " << j << ", " << k << "), naive_cpu_val: "
								<< grid_u1_cpu[index] << ", and fpga_val: " << grid_u1_d[index] << std::endl;
                		passed = false;
                	}
                }
            }
        }

        std::cout << "---------------------------------------------" << std::endl;
        std::cout << "               batch: " << bat << " ";

        if (passed)
        	std::cout << "Verification passed ";
        else
        	std::cout << "Verification failed ";

		std::cout << std::endl;
		std::cout << "---------------------------------------------" << std::endl;

    }
    std::cout << "============================================="  << std::endl << std::endl;
#endif

#ifdef DEBUG_VERBOSE
    std::cout << std::endl;
    std::cout << "*********************************************"  << std::endl;
    std::cout << "**      Debug info after calculations      **"  << std::endl;
    std::cout << "*********************************************"  << std::endl;

    for (unsigned int bat = 0; bat < gridData.batch; bat++)
    {
        int offset = bat * gridData.grid_size_x * gridData.grid_size_y * gridData.grid_size_y;

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
                    std::cout << "grid_id: (" << i << ", " << j << ", " << k << "), "
#ifndef FPGA_RUN_ONLY
                    		<< "golden_val: " << grid_u1_cpu[index]
#endif
							<< "fpga_explicit_val: " << grid_u1_d[index] << std::endl;
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

#ifndef FPGA_RUN_ONLY

	std::cout << " * naive stencil runtime  : " << runtime_init + runtime_naive_cpu_stencil<< " us" << std::endl;
	std::cout << "      |--> grid_init time : " << runtime_init << " us" << std::endl;
	std::cout << "      |--> calc time      : " << runtime_naive_cpu_stencil << " us" << std::endl;
#endif
	std::cout << " * fpga runtime           : " << runtime_init + h_to_d_runtime
				+ kernels_runtime + d_to_h_runtime << " us" << std::endl;
	std::cout << "      |--> grid_init time : " << runtime_init<< " us" << std::endl;
	std::cout << "      |--> h_to_d         : " << h_to_d_runtime << " us" << std::endl;
	std::cout << "      |--> d_to_h         : " << d_to_h_runtime << " us" << std::endl;
	std::cout << "      |--> kernels_runtime: " << kernels_runtime << " us" << std::endl;
	std::cout << "============================================="  << std::endl << std::endl;

    free(grid_u1_cpu);
    free(grid_u2_cpu);
    free(grid_u1_d);
    free(grid_u2_d);

	return 0;
}
