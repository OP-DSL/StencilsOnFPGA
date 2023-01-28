#include <ap_int.h>
#include <hls_stream.h>
#include <ap_axi_sdata.h>
#include <math.h>
#include <stdio.h>
#include "blacksholes_common.h"
#include "stencil.h"
#include "stencil.cpp"

// task level parallelism is applied using HLS data flow directive
// stream interface is used to connect the whole pipeline
void process_SLR(hls::stream <t_pkt> &in, hls::stream <t_pkt> &out,
		const int xdim0, const int size0, int size1, const int batches, const BlacksholesParameter computeParam){


    static hls::stream<uint256_dt> streamArray[SLR0_P_STAGE + 1];
    #pragma HLS STREAM variable = streamArray depth = 10

    struct data_G data_g;
    data_g.sizex = size0;
    data_g.sizey = size1;
    data_g.xdim0 = xdim0;
	data_g.end_index = (xdim0 >> SHIFT_BITS); // number of blocks with V number of elements to be processed in a single row
	data_g.end_row = size1; // includes the boundary
	data_g.outer_loop_limit = size1; // n + D/2
	data_g.gridsize = (data_g.end_row* batches + 1) * data_g.end_index;
	data_g.endindex_minus1 = data_g.end_index -1;
	data_g.endrow_plus1 = data_g.end_row + 1;
	data_g.endrow_plus2 = data_g.end_row + 2;
	data_g.endrow_minus1 = data_g.end_row - 1;
	data_g.total_itr_256 = data_g.end_row * data_g.end_index * batches;
	data_g.total_itr_512 = (data_g.end_row * data_g.end_index * batches + 1) >> 1;

	float alpha = computeParam.volatility * computeParam.volatility * computeParam.delta_t;
	float beta = computeParam.risk_free_rate * computeParam.delta_t;

	// parallel execution of following modules
	#pragma HLS dataflow

	axis2_fifo256(in, streamArray[0], data_g.total_itr_256);

	// unrolling iterative loop
	for(int i = 0; i < SLR0_P_STAGE; i++){
		#pragma HLS unroll
		process_grid(streamArray[i], streamArray[i+1], data_g, alpha, beta, computeParam.delta_t);
	}

	// sending data to kernel or mem2stream which resides in another SLR
	fifo256_2axis(streamArray[SLR0_P_STAGE], out, data_g.total_itr_256);
}

extern "C" {
	void stencil_SLR(
			const int size0,
			const int size1,
			const int xdim0,
			const int count,
			const int batches,
			const float spot_price,
			const float strike_price,
			const float time_to_maturity,
			const float volatility,
			const float risk_free_rate,
			const float delta_t,
			const float delta_S,
			const unsigned int N,
			const unsigned int K,
			const float SMaxFactor,
			hls::stream <t_pkt> &in,
			hls::stream <t_pkt> &out)
	{
			#pragma HLS INTERFACE s_axilite port = size0 bundle = control
			#pragma HLS INTERFACE s_axilite port = size1 bundle = control
			#pragma HLS INTERFACE s_axilite port = xdim0 bundle = control
			#pragma HLS INTERFACE s_axilite port = count bundle = control
			#pragma HLS INTERFACE s_axilite port = batches bundle = control

			#pragma HLS INTERFACE s_axilite port = spot_price  bundle = control
			#pragma HLS INTERFACE s_axilite port = strike_price  bundle = control
			#pragma HLS INTERFACE s_axilite port = time_to_maturity  bundle = control
			#pragma HLS INTERFACE s_axilite port = volatility  bundle = control
			#pragma HLS INTERFACE s_axilite port = risk_free_rate  bundle = control
			#pragma HLS INTERFACE s_axilite port = delta_t  bundle = control
			#pragma HLS INTERFACE s_axilite port = delta_S  bundle = control
			#pragma HLS INTERFACE s_axilite port = N  bundle = control
			#pragma HLS INTERFACE s_axilite port = K  bundle = control
			#pragma HLS INTERFACE s_axilite port = SMaxFactor  bundle = control

			#pragma HLS INTERFACE axis port = in  register
			#pragma HLS INTERFACE axis port = out register

			#pragma HLS INTERFACE s_axilite port = return bundle = control

			BlacksholesParameter computeParam;

			computeParam.spot_price = spot_price;
			computeParam.strike_price = strike_price;
			computeParam.time_to_maturity = time_to_maturity;
			computeParam.volatility = volatility;
			computeParam.risk_free_rate = risk_free_rate;
			computeParam.delta_t = delta_t;
			computeParam.delta_S = delta_S;
			computeParam.N = N;
			computeParam.K = K;
			computeParam.SMaxFactor = SMaxFactor;

			// iterative loop
			for(int i =  0; i < count*2; i++){
				process_SLR(in, out, xdim0, size0, size1, batches, computeParam);
			}

	}
}
