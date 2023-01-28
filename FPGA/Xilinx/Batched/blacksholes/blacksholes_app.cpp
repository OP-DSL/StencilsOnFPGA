
#include <iostream>
#include <cstdlib>
#include <chrono>
#include "xcl2.hpp"
#include "blacksholes_cpu.h"
#include "blacksholes_ops/blacksholes_ops.h"

//#define DEBUG_VERBOSE
#define MULTI_SLR
/******************************************************************************
* Main program
*******************************************************************************/
int main(int argc, char **argv)
{
	GridParameter gridProp;
	gridProp.logical_size_x = 200;
	gridProp.logical_size_y = 1;
	gridProp.batch = 10;
	gridProp.num_iter = 2040;

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

	//Allocating OPS instance
	OPS_instance * ops_inst = new OPS_instance(argc, argv, 1);

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


	double direct_calc_runtime = 0.0;

	for (int i = 0; i < gridProp.batch; i++)
	{
		double tmp_calc_runtime = 0.0;
		test_blacksholes_call_option(calcParam, &tmp_calc_runtime);
		direct_calc_runtime += tmp_calc_runtime;
	}

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
	float * grid_u3_cpu = (float*) aligned_alloc(4096, data_size_bytes);
	float * grid_u4_cpu = (float*) aligned_alloc(4096, data_size_bytes);
	float * grid_ops_result = (float*) aligned_alloc(4096, data_size_bytes);

	float * grid_u1_d = (float*) aligned_alloc(4096, data_size_bytes);
	float * grid_u2_d = (float*) aligned_alloc(4096, data_size_bytes);

	auto init_grid_start_point = std::chrono::high_resolution_clock::now();
	intialize_grid(grid_u1_cpu, gridProp, calcParam);
	auto init_grid_stop_copy_grid_start_point = std::chrono::high_resolution_clock::now();
	copy_grid(grid_u1_cpu, grid_u2_cpu, gridProp);
	auto copy_grid_stop_point = std::chrono::high_resolution_clock::now();
	copy_grid(grid_u1_cpu, grid_u3_cpu, gridProp);
	copy_grid(grid_u1_cpu, grid_u4_cpu, gridProp);
	copy_grid(grid_u1_cpu, grid_u1_d, gridProp);
//	copy_grid(grid_u1_cpu, grid_u1_ops, gridProp);
//	copy_grid(grid_u1_cpu, grid_u2_ops, gridProp);

	double init_grid_runtime = std::chrono::duration<double, std::micro>
				(init_grid_stop_copy_grid_start_point - init_grid_start_point).count();
	double copy_grid_runtime = std::chrono::duration<double, std::micro>
				(copy_grid_stop_point - init_grid_stop_copy_grid_start_point).count();

	//golden stencil computation on host
	auto explicit1_start_point = std::chrono::high_resolution_clock::now();
	bs_explicit1(grid_u1_cpu, grid_u2_cpu, gridProp, calcParam);
	auto explicit1_stop_explicit2_start_point = std::chrono::high_resolution_clock::now();
	bs_explicit2(grid_u3_cpu, grid_u4_cpu, gridProp, calcParam);
	auto explicit2_stop_point_explicit_ops_start_point = std::chrono::high_resolution_clock::now();

	//ops implementation
	bs_explicit1_ops(grid_ops_result, ops_inst, gridProp, calcParam);
	auto explicit_ops_stop_point = std::chrono::high_resolution_clock::now();

	double explicit1_runtime = std::chrono::duration<double, std::micro>
				(explicit1_stop_explicit2_start_point - explicit1_start_point).count();
	double explicit2_runtime = std::chrono::duration<double, std::micro>
				(explicit2_stop_point_explicit_ops_start_point - explicit1_stop_explicit2_start_point).count();
	double explicit_ops_runtime = std::chrono::duration<double, std::micro>
				(explicit_ops_stop_point - explicit2_stop_point_explicit_ops_start_point).count();

	//OPENCL HOST CODE AREA START
	auto binaryFile = argv[1];
	cl_int err;
	cl::Event event;

	auto devices = xcl::get_devices("Xilinx");
	auto device = devices[0];

	OCL_CHECK(err, cl::Context context(device, NULL, NULL, NULL, &err));
	OCL_CHECK(err, cl::CommandQueue queue(context, device, CL_QUEUE_PROFILING_ENABLE | CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, &err));
	OCL_CHECK(err, std::string device_name = device.getInfo<CL_DEVICE_NAME>(&err));

	//Create Program and Kernel
	auto fileBuf = xcl::read_binary_file(binaryFile);
	cl::Program::Binaries bins{{fileBuf.data(), fileBuf.size()}};

	OCL_CHECK(err, cl::Program program(context, {device}, bins, NULL, &err))
	OCL_CHECK(err, cl::Kernel krnl_slr0(program, "stencil_SLR", &err));
#ifdef MULTI_SLR
	OCL_CHECK(err, cl::Kernel krnl_slr1(program, "stencil_SLR", &err));
	OCL_CHECK(err, cl::Kernel krnl_slr2(program, "stencil_SLR", &err));
#endif
	OCL_CHECK(err, cl::Kernel krnl_mem2stream(program, "stencil_mem2stream", &err));

	//Allocation Buffer in Global Memory
	OCL_CHECK(err, cl::Buffer buffer_curr(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, data_size_bytes, grid_u1_d, &err));
	OCL_CHECK(err, cl::Buffer buffer_next(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, data_size_bytes, grid_u2_d, &err));

#ifdef MULTI_SLR
	unsigned int total_SLR = 3;
#else
	unsigned int total_SLR = 1;
#endif

	unsigned int number_of_process_grid_per_SLR = 4;
	unsigned int total_process_grids_per_iter =  total_SLR * number_of_process_grid_per_SLR * 2;
	unsigned int num_iter = gridProp.num_iter/total_process_grids_per_iter;
	//set Kernel arguments
	int narg = 0;
	OCL_CHECK(err, err = krnl_slr0.setArg(narg++, gridProp.logical_size_x));
	OCL_CHECK(err, err = krnl_slr0.setArg(narg++, gridProp.logical_size_y));
	OCL_CHECK(err, err = krnl_slr0.setArg(narg++, gridProp.grid_size_x));
	OCL_CHECK(err, err = krnl_slr0.setArg(narg++, num_iter));
	OCL_CHECK(err, err = krnl_slr0.setArg(narg++, gridProp.batch));
	OCL_CHECK(err, err = krnl_slr0.setArg(narg++, calcParam.spot_price));
	OCL_CHECK(err, err = krnl_slr0.setArg(narg++, calcParam.strike_price));
	OCL_CHECK(err, err = krnl_slr0.setArg(narg++, calcParam.time_to_maturity));
	OCL_CHECK(err, err = krnl_slr0.setArg(narg++, calcParam.volatility));
	OCL_CHECK(err, err = krnl_slr0.setArg(narg++, calcParam.risk_free_rate));
	OCL_CHECK(err, err = krnl_slr0.setArg(narg++, calcParam.delta_t));
	OCL_CHECK(err, err = krnl_slr0.setArg(narg++, calcParam.delta_S));
	OCL_CHECK(err, err = krnl_slr0.setArg(narg++, calcParam.N));
	OCL_CHECK(err, err = krnl_slr0.setArg(narg++, calcParam.K));
	OCL_CHECK(err, err = krnl_slr0.setArg(narg++, calcParam.SMaxFactor));

#ifdef MULTI_SLR
	narg = 0;
	OCL_CHECK(err, err = krnl_slr1.setArg(narg++, gridProp.logical_size_x));
	OCL_CHECK(err, err = krnl_slr1.setArg(narg++, gridProp.logical_size_y));
	OCL_CHECK(err, err = krnl_slr1.setArg(narg++, gridProp.grid_size_x));
	OCL_CHECK(err, err = krnl_slr1.setArg(narg++, num_iter));
	OCL_CHECK(err, err = krnl_slr1.setArg(narg++, gridProp.batch));
	OCL_CHECK(err, err = krnl_slr1.setArg(narg++, calcParam.spot_price));
	OCL_CHECK(err, err = krnl_slr1.setArg(narg++, calcParam.strike_price));
	OCL_CHECK(err, err = krnl_slr1.setArg(narg++, calcParam.time_to_maturity));
	OCL_CHECK(err, err = krnl_slr1.setArg(narg++, calcParam.volatility));
	OCL_CHECK(err, err = krnl_slr1.setArg(narg++, calcParam.risk_free_rate));
	OCL_CHECK(err, err = krnl_slr1.setArg(narg++, calcParam.delta_t));
	OCL_CHECK(err, err = krnl_slr1.setArg(narg++, calcParam.delta_S));
	OCL_CHECK(err, err = krnl_slr1.setArg(narg++, calcParam.N));
	OCL_CHECK(err, err = krnl_slr1.setArg(narg++, calcParam.K));
	OCL_CHECK(err, err = krnl_slr1.setArg(narg++, calcParam.SMaxFactor));

	narg = 0;
	OCL_CHECK(err, err = krnl_slr2.setArg(narg++, gridProp.logical_size_x));
	OCL_CHECK(err, err = krnl_slr2.setArg(narg++, gridProp.logical_size_y));
	OCL_CHECK(err, err = krnl_slr2.setArg(narg++, gridProp.grid_size_x));
	OCL_CHECK(err, err = krnl_slr2.setArg(narg++, num_iter));
	OCL_CHECK(err, err = krnl_slr2.setArg(narg++, gridProp.batch));
	OCL_CHECK(err, err = krnl_slr2.setArg(narg++, calcParam.spot_price));
	OCL_CHECK(err, err = krnl_slr2.setArg(narg++, calcParam.strike_price));
	OCL_CHECK(err, err = krnl_slr2.setArg(narg++, calcParam.time_to_maturity));
	OCL_CHECK(err, err = krnl_slr2.setArg(narg++, calcParam.volatility));
	OCL_CHECK(err, err = krnl_slr2.setArg(narg++, calcParam.risk_free_rate));
	OCL_CHECK(err, err = krnl_slr2.setArg(narg++, calcParam.delta_t));
	OCL_CHECK(err, err = krnl_slr2.setArg(narg++, calcParam.delta_S));
	OCL_CHECK(err, err = krnl_slr2.setArg(narg++, calcParam.N));
	OCL_CHECK(err, err = krnl_slr2.setArg(narg++, calcParam.K));
	OCL_CHECK(err, err = krnl_slr2.setArg(narg++, calcParam.SMaxFactor));
#endif

	narg = 0;
	OCL_CHECK(err, err = krnl_mem2stream.setArg(narg++, buffer_curr));
	OCL_CHECK(err, err = krnl_mem2stream.setArg(narg++, buffer_next));
	OCL_CHECK(err, err = krnl_mem2stream.setArg(narg++, num_iter));
	OCL_CHECK(err, err = krnl_mem2stream.setArg(narg++, gridProp.grid_size_x));
	OCL_CHECK(err, err = krnl_mem2stream.setArg(narg++, gridProp.grid_size_y));
	OCL_CHECK(err, err = krnl_mem2stream.setArg(narg++, gridProp.batch));

	//Copy input buffer to device
	auto h_to_d_start_point = std::chrono::high_resolution_clock::now();
	OCL_CHECK(err, err = queue.enqueueMigrateMemObjects({buffer_curr, buffer_next}, 0));
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
	OCL_CHECK(err, err = queue.enqueueMigrateMemObjects({buffer_curr}, CL_MIGRATE_MEM_OBJECT_HOST));
	queue.finish();
	auto d_to_h_stop_point = std::chrono::high_resolution_clock::now();

	double h_to_d_runtime = std::chrono::duration<double, std::micro>
				(h_to_d_stop_kernels_start_point - h_to_d_start_point).count();
	double kernels_runtime = std::chrono::duration<double, std::micro>
				(kernels_stop_d_to_h_start_point - h_to_d_stop_kernels_start_point).count();
	double d_to_h_runtime = std::chrono::duration<double, std::micro>
				(d_to_h_stop_point - kernels_stop_d_to_h_start_point).count();

//	for (int i = 0; i < gridProp.act_size_x; i++)
//	{
//		if (abs(grid_u1_cpu[i] - grid_ops_result[i]) > EPSILON)
//		{
//			std::cout << "value mismatch. i: " << i << " cpu: " << grid_u2_cpu[i] << " ops: " << grid_ops_result[i] << std::endl;
//		}
//		else
//		{
//			std::cout << "i: " << i << " cpu: " << grid_u2_cpu[i] << " ops: " << grid_ops_result[i] << std::endl;
//		}
//	}

//	for (int i = 0; i < gridProp.logical_size_x; i++)
//	{
//		std::cout << "idx: " << i << ", dat_current: " << grid_ops_result[i] << std::endl;
//	}
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
					 << " explicit1_ops_val: " << grid_ops_result[offset + i]
					 << " istvan explicit val: " << grid_u3_cpu[offset + i]
					 << " fpga_explicit_val: " << grid_u1_d[offset + i] << std::endl;
		}
	}
#endif
	
	std::cout << "call option price from explicit method: " << get_call_option(grid_u1_cpu, gridProp, calcParam) << std::endl;
	std::cout << "call option price from istvan explicit method: " << get_call_option(grid_u3_cpu, gridProp, calcParam) << std::endl;
	std::cout << "call option price from ops explicit method: " << get_call_option(grid_ops_result, gridProp, calcParam) << std::endl;
	std::cout << "call option price from fpga explicit method: " << get_call_option(grid_u1_d, gridProp, calcParam) << std::endl;

	std::cout << "============================================="  << std::endl << std::endl;


	std::cout << std::endl;
	std::cout << "*********************************************"  << std::endl;
	std::cout << "**            runtime summery              **"  << std::endl;
	std::cout << "*********************************************"  << std::endl;

	std::cout << " * direct_cal time        : " << direct_calc_runtime << " us" << std::endl;
	std::cout << "      |--> time_per_option: " << direct_calc_runtime  / gridProp.batch << " us" << std::endl;
	std::cout << " * naive stencil runtime  : " << init_grid_runtime + copy_grid_runtime + explicit1_runtime << " us" << std::endl;
	std::cout << "      |--> grid_init time : " << init_grid_runtime + copy_grid_runtime << " us" << std::endl;
	std::cout << "      |--> calc time      : " << explicit1_runtime << " us" << std::endl;
	std::cout << " * naive stencil2 runtime : " << init_grid_runtime + copy_grid_runtime + explicit2_runtime << " us" << std::endl;
	std::cout << "      |--> grid_init time : " << init_grid_runtime + copy_grid_runtime << " us" << std::endl;
	std::cout << "      |--> calc time      : " << explicit2_runtime << " us" << std::endl;
	std::cout << " * ops stencil runtime    : " << explicit_ops_runtime << " us" << std::endl;
	std::cout << " * fpga runtime           : " << init_grid_runtime + h_to_d_runtime
				+ kernels_runtime + d_to_h_runtime << " us" << std::endl;
	std::cout << "      |--> grid_init time : " << init_grid_runtime<< " us" << std::endl;
	std::cout << "      |--> h_to_d         : " << h_to_d_runtime << " us" << std::endl;
	std::cout << "      |--> d_to_h         : " << d_to_h_runtime << " us" << std::endl;
	std::cout << "      |--> kernels_runtime: " << kernels_runtime << " us" << std::endl;
	std::cout << "============================================="  << std::endl << std::endl;
	//ops_exit
	delete ops_inst;

	//Free memory
	free(grid_u1_cpu);
	free(grid_u2_cpu);
	free(grid_u3_cpu);
	free(grid_u4_cpu);
	free(grid_ops_result);
	free(grid_u1_d);
	free(grid_u2_d);

	return 0;
}


