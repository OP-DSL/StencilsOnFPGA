#include "stencil.h"

#define OPTIMIZED_REDUCTION
#define DEBUG_VERBOSE

template <typename T>
static T register_it(T x)
{
#pragma HLS inline off
	T temp = x;
	return temp;
}

static void axis2_fifo256(hls::stream <t_pkt> &in, hls::stream<uint256_dt> &out,  const unsigned int total_itr)
{
	for (int itr = 0; itr < total_itr; itr++){
		#pragma HLS PIPELINE II=1
		#pragma HLS loop_tripcount min=min_grid max=max_grid avg=avg_grid
		t_pkt tmp = in.read();
//		printf("reading 256 bit input from SLR. iter: %d\n", itr);
		out << tmp.data;
	}
}

static void fifo256_2axis(hls::stream <uint256_dt> &in, hls::stream<t_pkt> &out, const unsigned int total_itr)
{
	for (int itr = 0; itr < total_itr; itr++){
		#pragma HLS PIPELINE II=1
		#pragma HLS loop_tripcount min=min_grid max=max_grid avg=avg_grid
		t_pkt tmp;
		tmp.data = in.read();
		out.write(tmp);
	}
}

static void inline get_stencil_coefficent(const float & alpha, const float & beta,
		const float & delta_t, const int & init_idx, float * a,
		float * b, float * c)
{
	for (int i = 0; i < VEC_FACTOR; i++)
	{
#pragma HLS UNROLL

		int idx = init_idx + i;
		a[i] = 0.5 * (alpha * std::pow(idx,2) - beta * idx);
		b[i] = 1 - alpha * std::pow(idx,2) - beta;
		c[i] = 0.5 * (alpha * std::pow(idx,2) + beta * idx);
	}
}

static void process_grid(hls::stream<uint256_dt> &rd_buffer, hls::stream<uint256_dt> &wr_buffer, struct data_G data_g,
		const float coefficients[7])
{
	float s_1_1_2_arr[VEC_FACTOR];
	float s_1_2_1_arr[VEC_FACTOR];
	float s_1_1_1_arr[VEC_FACTOR+2];
	float s_1_0_1_arr[VEC_FACTOR];
	float s_1_1_0_arr[VEC_FACTOR];

	float mem_wr[VEC_FACTOR];

#pragma HLS ARRAY_PARTITION variable=s_1_1_2_arr complete dim=1
#pragma HLS ARRAY_PARTITION variable=s_1_2_1_arr complete dim=1
#pragma HLS ARRAY_PARTITION variable=s_1_1_1_arr complete dim=1
#pragma HLS ARRAY_PARTITION variable=s_1_0_1_arr complete dim=1
#pragma HLS ARRAY_PARTITION variable=s_1_1_0_arr complete dim=1
#pragma HLS ARRAY_PARTITION variable=mem_wr complete dim=1

	uint256_dt windowBuff_1[max_depth_xy];
	uint256_dt windowBuff_2[max_depth_8];
	uint256_dt windowBuff_3[max_depth_8];
	uint256_dt windowBuff_4[max_depth_xy];

#pragma HLS BIND_STORAGE variable=windowBuff_1 type=ram_t2p impl=uram latency=1
#pragma HLS BIND_STORAGE variable=windowBuff_2 type=ram_t2p impl=uram latency=1
#pragma HLS BIND_STORAGE variable=windowBuff_3 type=ram_t2p impl=uram latency=1
#pragma HLS BIND_STORAGE variable=windowBuff_4 type=ram_t2p impl=uram latency=1

	uint256_dt s_1_1_2, s_1_2_1, s_1_1_1, s_1_1_1b, s_1_1_1f, s_1_0_1, s_1_1_0;
	uint256_dt update_j;

	unsigned short i = 0, j = 0, k = 0;
	unsigned short j_p = 0, j_l = 0;

	for(unsigned int itr = 0; itr < data_g.gridsize_pr; itr++)
	{
#pragma HLS loop_tripcount min=min_grid max=max_grid avg=avg_grid
#pragma HLS PIPELINE II=1

		spc_temp_blocking_read:
		{
			bool cond_x = (i == data_g.xblocks);
			bool cond_y = (j == data_g.grid_size_y - 1);
			bool cond_z = (k == data_g.limit_z - 1);

			if (cond_x)
				i = 0;

			if (cond_y && cond_x)
				j = 0;
			else if(cond_x)
				j++;

			if (cond_x && cond_y && cond_z)
				k = 1;
			else if(cond_y && cond_x)
				k++;

			s_1_1_0 = windowBuff_4[j_p];

			s_1_0_1 = windowBuff_3[j_l];
			windowBuff_4[j_p] = s_1_0_1;

			s_1_1_1b = s_1_1_1;
			windowBuff_3[j_l] = s_1_1_1b;

			s_1_1_1 = s_1_1_1f;
			s_1_1_1f = windowBuff_2[j_l];

			s_1_2_1 = windowBuff_1[j_p];
			windowBuff_2[j_l] = s_1_2_1;

			bool cond_read = (itr < data_g.gridsize_da);

			if (cond_read)
			{
//				printf("reading data for iteration %d\n", itr);
				s_1_1_2 = rd_buffer.read();
			}

			windowBuff_1[j_p] = s_1_1_2;

			bool cond_eo_plane = (j_p == data_g.plane_diff);
			bool cond_eo_line = (j_l == data_g.line_diff);

			if (cond_eo_plane)
				j_p = 0;
			else
				j_p++;

			if(cond_eo_line)
				j_l = 0;
			else
				j_l++;
		}

		vec2arr:
		{
			for (int id = 0; id < VEC_FACTOR; id++)

			{
#pragma HLS UNROLL

				data_conv s_1_1_2_u, s_1_2_1_u, s_1_1_1_u, s_1_0_1_u, s_1_1_0_u;

				s_1_1_2_u.i = s_1_1_2.range(DATATYPE_SIZE * (id + 1) - 1, id * DATATYPE_SIZE);
				s_1_2_1_u.i = s_1_2_1.range(DATATYPE_SIZE * (id + 1) - 1, id * DATATYPE_SIZE);
				s_1_1_1_u.i = s_1_1_1.range(DATATYPE_SIZE * (id + 1) - 1, id * DATATYPE_SIZE);
				s_1_0_1_u.i = s_1_0_1.range(DATATYPE_SIZE * (id + 1) - 1, id * DATATYPE_SIZE);
				s_1_1_0_u.i = s_1_1_0.range(DATATYPE_SIZE * (id + 1) - 1, id * DATATYPE_SIZE);

				s_1_1_2_arr[id]   =  s_1_1_2_u.f;
				s_1_2_1_arr[id]   =  s_1_2_1_u.f;
				s_1_1_1_arr[id+1] =  s_1_1_1_u.f;
				s_1_0_1_arr[id]   =  s_1_0_1_u.f;
				s_1_1_0_arr[id]   =  s_1_1_0_u.f;
			}

			data_conv tmp1_o1, tmp2_o2;
			tmp1_o1.i = s_1_1_1b.range(DATATYPE_SIZE * (VEC_FACTOR) - 1, (VEC_FACTOR-1) * DATATYPE_SIZE);
			tmp2_o2.i = s_1_1_1f.range(DATATYPE_SIZE * (0 + 1) - 1, 0 * DATATYPE_SIZE);
			s_1_1_1_arr[0] = tmp1_o1.f;
			s_1_1_1_arr[VEC_FACTOR + 1] = tmp2_o2.f;
		}

		process:
		{
			unsigned short y_index = j + data_g.offset_y;

			for (short q = 0; q < VEC_FACTOR; q++)
			{
#pragma HLS UNROLL

				short index = (i << SHIFT_BITS) + q + data_g.offset_x;

				float r1_1_2 =  s_1_1_2_arr[q] * coefficients[0];
				float r1_2_1 =  s_1_2_1_arr[q] * coefficients[1];
				float r0_1_1 =  s_1_1_1_arr[q] * coefficients[2];
				float r1_1_1 =  s_1_1_1_arr[q+1] * coefficients[3];
				float r2_1_1 =  s_1_1_1_arr[q+2] * coefficients[4];
				float r1_0_1 =  s_1_0_1_arr[q] * coefficients[5];
				float r1_1_0 =  s_1_1_0_arr[q] * coefficients[6];

#ifdef OPTIMIZED_REDUCTION
				float f1 = r1_1_2 + r1_2_1;
				float f2 = r0_1_1 + r1_1_1;
				float f3 = r2_1_1 + r1_0_1;

#pragma HLS BIND_OP variable=f1 op=fadd
#pragma HLS BIND_OP variable=f2 op=fadd

				float r1 = f1 + f2;
				float r2 = f3 + r1_1_0;

				float result = r1 + r2;
#else
				float result = r1_1_2 + r1_2_1 + r0_1_1 + r1_1_1 + r2_1_1 + r1_0_1 + r1_1_0;
#endif

				bool cond_change = register_it <bool> (index <= data_g.offset_x || index > data_g.sizex
						|| (k <= 1) || (k >= data_g.limit_z -1) || (y_index <= 0) || (y_index >= data_g.grid_size_y - 1));

				mem_wr[q] = cond_change ? s_1_1_1[q+1] : result;
			}
		}

		array2vec: for (int q = 0; q < VEC_FACTOR; q++)
		{
#pragma HLS UNROLL
			data_conv tmp;
			tmp.f = mem_wr[q];
			update_j.range(DATATYPE_SIZE * (q + 1) - 1, q * DATATYPE_SIZE) = tmp.i;
		}

		write:
		{
			bool cond_wr = (k >= 1) && (k < data_g.limit_z);

			if (cond_wr)
				wr_buffer << update_j;
		}

		// move cell block
		i++;

	}

}
