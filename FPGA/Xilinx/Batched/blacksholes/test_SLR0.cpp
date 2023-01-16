 
#include "SLR0.cpp"
#include "blacksholes_cpu.h"
#include "blacksholes_cpu.cpp"
#include <iostream>

int main()
{
	GridParameter gridProp;
	gridProp.logical_size_x = 200;
	gridProp.logical_size_y = 1;
	gridProp.batch = 1;
	gridProp.num_iter = 1; //test_interation
	gridProp.act_size_x = gridProp.logical_size_x+2;
	gridProp.act_size_y = 1;
	unsigned int vectorization_factor = VEC_FACTOR;
	//padding each row as multiple of vectorization factor
	gridProp.grid_size_x = (gridProp.act_size_x % vectorization_factor) != 0 ?
			(gridProp.act_size_x/vectorization_factor + 1) * vectorization_factor :
			gridProp.act_size_x;
	gridProp.grid_size_y = gridProp.act_size_y;
	unsigned int data_size = gridProp.grid_size_x * gridProp.batch; //1D grid
	unsigned int data_size_bytes = gridProp.grid_size_x * gridProp.grid_size_y * sizeof(float) * gridProp.batch;

	data_G data_g;
	data_g.sizex = gridProp.logical_size_x;
	data_g.sizey = gridProp.logical_size_y; //since 1D grid
	data_g.xdim0 = gridProp.grid_size_x;
	data_g.end_index = gridProp.grid_size_x >> SHIFT_BITS;
	data_g.end_row = 1; //since 1D grid
	data_g.outer_loop_limit = 1; //since 1D grid
	data_g.gridsize = (data_g.end_row * gridProp.batch + 1) * data_g.end_index;
	data_g.endindex_minus1 = data_g.end_index - 1;
	data_g.endrow_plus1 = data_g.end_row + 1;
	data_g.endrow_plus2 = data_g.end_row + 2;
	data_g.endrow_minus1 = data_g.end_row - 1;
	data_g.total_itr_256 = data_g.end_row * data_g.end_index * gridProp.batch;
	data_g.total_itr_512 = (data_g.total_itr_256 + 1) >> 1;

	BlacksholesParameter calcParam;
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

	float alpha = calcParam.volatility * calcParam.volatility * calcParam.delta_t;
	float beta = calcParam.risk_free_rate * calcParam.delta_t;

	float data[data_size];
	float data_cpu_current[data_size];
	float data_cpu_next[data_size];

	//generate data
	intialize_grid(data, gridProp, calcParam);
	copy_grid(data, data_cpu_current, gridProp);
	copy_grid(data, data_cpu_next, gridProp);

	bs_explicit1(data_cpu_current, data_cpu_next, gridProp, calcParam);

	uint512_dt * data_current = (uint512_dt*)aligned_alloc(4096, data_size_bytes);
	uint512_dt * data_next = (uint512_dt*)aligned_alloc(4096, data_size_bytes);

	for (int i = 0; i < data_g.total_itr_512; i++)
	{
		uint256_dt tmp_vec;
		std::cout << "vector written: " << i << "={";

		for (int j = 0; j < VEC_FACTOR * 2; j++)
		{
			unsigned int index = i * (VEC_FACTOR * 2) + j;
			data_conv tmp;
			tmp.f = data[index];

            if (j != (VEC_FACTOR * 2) - 1)
            	std::cout << tmp.f << ", ";
            else
            	std::cout << tmp.f <<"}" << std::endl;

            tmp_vec.range(DATATYPE_SIZE * (j + 1) - 1, j * DATATYPE_SIZE) = tmp.i;
		}

		data_current[i] = tmp_vec;
		data_next[i] = tmp_vec;
	}

	hls::stream<t_pkt>  fallback_stream;

	const int size0 = gridProp.logical_size_x;
	const int size1 = gridProp.logical_size_y;
	const int xdim0 = gridProp.grid_size_x;
	const int count = gridProp.num_iter;
	const int batches = gridProp.batch;

	stencil_SLR0(data_current, data_next, size0, size1, xdim0,
			count, batches, calcParam, fallback_stream, fallback_stream);

	for (int i = 0; i < data_g.total_itr_512; i++)
	{
		uint512_dt input_vec;
		input_vec = data_next[i];
		std::cout << "hls vector result: " << i << "={";
		for (int j = 0; j < VEC_FACTOR * 2; j++)
		{
			unsigned int index = i * (VEC_FACTOR * 2) + j;
			data_conv tmp;
			tmp.i = input_vec.range(DATATYPE_SIZE * (j + 1) - 1, j * DATATYPE_SIZE);

            if (j != (VEC_FACTOR * 2) - 1)
            	std::cout << tmp.f << ", ";
            else
            	std::cout << tmp.f <<"}\t";
		}

		std::cout << "cpu vector result: " << i << "={";
		for (int j = 0; j < VEC_FACTOR * 2; j++)
		{
			unsigned int index = i * (VEC_FACTOR * 2) + j;

            if (j != (VEC_FACTOR * 2) - 1)
            	std::cout << data_cpu_next[index] << ", ";
            else
            	std::cout << data_cpu_next[index] <<"}" << std::endl;
		}
	}

//	Test process grid
//
//	hls::stream<uint256_dt> in;
//	hls::stream<uint256_dt> out;

//	for (int i = 0; i < data_g.total_itr_256; i++)
//	{
//		uint256_dt output_vec;
//		std::cout << "vector written: " << i << "={";
//
//		for (int j = 0; j < VEC_FACTOR; j++)
//		{
//			unsigned int index = i*VEC_FACTOR + j;
//			data_conv tmp;
//            tmp.f = data[index];
//
//            if (j != VEC_FACTOR - 1)
//            	std::cout << tmp.f << ", ";
//            else
//            	std::cout << tmp.f <<"}" << std::endl;
//
//            output_vec.range(DATATYPE_SIZE * (j + 1) - 1, j * DATATYPE_SIZE) = tmp.i;
//		}
//		in << output_vec;
//	}
//
//	process_grid(in, out, data_g, alpha, beta, calcParam.delta_t);
//
//	for (int i = 0; i < data_g.total_itr_256; i++)
//	{
//		uint256_dt input_vec;
//		input_vec = out.read();
//		std::cout << "hls vector result: " << i << "={";
//		for (int j = 0; j < VEC_FACTOR; j++)
//		{
//			unsigned int index = i*VEC_FACTOR + j;
//			data_conv tmp;
//			tmp.i = input_vec.range(DATATYPE_SIZE * (j + 1) - 1, j * DATATYPE_SIZE);
//
//            if (j != VEC_FACTOR - 1)
//            	std::cout << tmp.f << ", ";
//            else
//            	std::cout << tmp.f <<"}\t";
//		}
//
//		std::cout << "cpu vector result: " << i << "={";
//		for (int j = 0; j < VEC_FACTOR; j++)
//		{
//			unsigned int index = i*VEC_FACTOR + j;
//
//            if (j != VEC_FACTOR - 1)
//            	std::cout << data_cpu_next[index] << ", ";
//            else
//            	std::cout << data_cpu_next[index] <<"}" << std::endl;
//		}
//	}


	return 0;
}

