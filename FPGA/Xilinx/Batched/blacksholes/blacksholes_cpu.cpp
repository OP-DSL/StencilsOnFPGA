#include "blacksholes_cpu.h"


float standard_normal_CDF(float val)
{
	return 0.5 * erfc(-val * INV_SQRT_2);
}

float blacksholes_call_option(float spot_price, float strike_price,
		float time_to_maturity, float risk_free_interest_rate, float volatility)
{
	float d1 = (log(spot_price / strike_price) + (risk_free_interest_rate + pow(volatility,2) / 2) * time_to_maturity)
							/ (volatility * sqrt(time_to_maturity));

	float d2 = d1 - volatility * sqrt(time_to_maturity);
	float return_on_portfolio = standard_normal_CDF(d1) * spot_price;
	float return_on_deposit = standard_normal_CDF(d2) * strike_price * exp(-risk_free_interest_rate * time_to_maturity);

	return return_on_portfolio - return_on_deposit;
}

void test_blacksholes_call_option()
{
	//testing blacksholes_call_option

//	float spot_price = 62;
//	float strike_price = 60;
//	float t = 40.0/365.0; //40 days
//	float volatility = 0.32; //32%
//	float risk_free_rate = 0.04; //4%
	float spot_price = 16;
	float strike_price = 10;
	float t = 0.25;
	float volatility = 0.4; //40%
	float risk_free_rate = 0.1; //10%

	std::cout << "*********************************************"  << std::endl;
	std::cout << "** testing blacksholes direct calculations **"  << std::endl;
	std::cout << "*********************************************"  << std::endl;

	std::cout << "spot_price		: " << spot_price << std::endl;
	std::cout << "strike_price		: " << strike_price << std::endl;
	std::cout << "time_to_maturity	: " << t << std::endl;
	std::cout << "volatility		: " << volatility << std::endl;
	std::cout << "risk_free_rate	: " << risk_free_rate << std::endl;
	std::cout << std::endl;
	float option_price = blacksholes_call_option(spot_price, strike_price, t, risk_free_rate, volatility);
	std::cout << "option_price 		: " << option_price << std::endl;

//	float diff = fabs(option_price - 3.85894);
	float diff = fabs(option_price - 6.252287);

	assert(diff < EPSILON);
}


// golden non optimised stencil computation on host PC. 1D stencil calculation
int bs_implicit1(float* current, float *next, GridParameter gridData, BlacksholesParameter computeParam)
{
	for (unsigned int bat = 0; bat < gridData.batch; bat++)
	{
		int offset 	= bat * gridData.grid_size_x;
		float alpha = computeParam.volatility * computeParam.volatility * computeParam.delta_t;
		float beta = computeParam.risk_free_rate * computeParam.delta_t;

		float ak[gridData.grid_size_x];
		float bk[gridData.grid_size_x];
		float ck[gridData.grid_size_x];

		for (unsigned int i = 0; i < computeParam.N; i+=2)
		{
			for (unsigned int j = 1; j < gridData.act_size_x - 1; j++) //excluding ghost
			{
				if (i == 0) //Initializing coefficients
				{
					unsigned int index = j;
					ak[j] = 0.5 * (alpha * index * index - beta * index);
					bk[j] = 1 - alpha * index * index - beta;
					ck[j] = 0.5 * (alpha * index * index + beta * index);
				}

				next[offset + j] = ak[j] * current[offset + j - 1]
								 + bk[j] * current[offset + j]
								 + ck[j] * current[offset + j + 1];
			}

			for (unsigned int j = 1; j < gridData.act_size_x - 1; j++) //excluding ghost
			{
				current[offset + j]	= ak[j] * next[offset + j - 1]
									+ bk[j] * next[offset + j]
									+ ck[j] * next[offset + j + 1];

//				std::cout << "grid_id: " << j << " ak: " << ak[j] << " bk: " << bk[j] << " ck: " << ck[j] << " current(j-1): " << current[offset + (j - 1)]
//						<< " current(j): " << 	current[offset + j] << " current(j+1): "	<< current[offset + (j + 1)] << std::endl;
			}
		}
	}
	return 0;
}

//get the exact call option pricing for given spot price and strike price
float get_call_option(float* current, GridParameter gridData, BlacksholesParameter computeParam)
{
	float index 	= (float)computeParam.spot_price / ((float) computeParam.strike_price * computeParam.SMaxFactor) * computeParam.K;
	unsigned int indexLower 	= (int)std::floor(index);
	unsigned int indexUpper 	= indexLower + 1;

	float option_price = 0.0;

	if (indexUpper < computeParam.K)
		option_price = (current[indexLower] * (indexUpper - index) + current[indexUpper] * (index - indexLower));
	else
		option_price = current[computeParam.K];

	return option_price;
}



int bs_implicit_istvan(float* current, float *next, GridParameter gridData, BlacksholesParameter computeParam)
{
	computeParam.delta_S = computeParam.SMaxFactor * computeParam.strike_price / (computeParam.K - 1); //This is how it is defined in istvan' implementation.
	float c1 = 0.5 * computeParam.delta_t * computeParam.volatility * computeParam.volatility / (computeParam.delta_S * computeParam.delta_S);
	float c2 = 0.5 * computeParam.delta_t * computeParam.risk_free_rate / computeParam.delta_S;
	float c3 = computeParam.risk_free_rate * computeParam.delta_t;
	float S, lambda, gamma;
	float a[gridData.grid_size_x], b[gridData.grid_size_x], c[gridData.grid_size_x];

	for (unsigned int bat = 0; bat < gridData.batch; bat++)
	{
		unsigned int offset = bat * gridData.grid_size_x;

		//intialize data
		current[offset + 0] = 0.0f;
		next[offset + 0] = 0.0f;

//		std::cout << "Init curent[" << 0 << "]: " << current[offset + 0]  << std::endl;

		for (unsigned int i = 0; i < gridData.act_size_x - 2; i++)
		{
			current[offset + i+1] = (i*computeParam.delta_S) > computeParam.strike_price ? (i*computeParam.delta_S - computeParam.strike_price) : 0.0f;
//			std::cout << "Init curent[" << i + 1 << "]: " << current[offset + i + 1]  << std::endl;
		}

		current[offset + gridData.act_size_x - 1] = 0.0f;
		next[offset + gridData.act_size_x - 1] = 0.0f;
//		std::cout << "Init curent[" << gridData.act_size_x - 1 << "]: " << current[offset + gridData.act_size_x - 1]  << std::endl;

		for (unsigned int i = 0; i < computeParam.N; i+=2)
		{
			for (unsigned int j = 1; j < gridData.act_size_x - 1; j++) //excluding ghost
			{
				if (i == 0) //calculating coefficients
				{
					S = (j - 1) * computeParam.delta_S;
					lambda = c1 * S * S;
					gamma = c2 * S;

					if (j == gridData.act_size_x - 1)
					{
						a[j] = - 2.0f * gamma;
						b[j] = + 2.0f * gamma - c3;
						c[j] = 0.0f;
					}
					else
					{
						a[j] = lambda - gamma;
						b[j] = - 2.0f * lambda - c3;
						c[j] = lambda + gamma;
					}
				}

				next[offset + j] = current[offset + j]
								 + a[j]*current[offset + j - 1]
								 + b[j]*current[offset + j]
								 + c[j]*current[offset + j + 1];
			}

			for (unsigned int j = 1; j < gridData.act_size_x - 1; j++)
			{
				current[offset + j] = next[j]
								    + a[j]*next[offset + j - 1]
									+ b[j]*next[offset + j]
									+ c[j]*next[offset + j + 1];
			}
		}
	}

	return 0;
}

bool stencil_stability(BlacksholesParameter computeParam)
{
	std::cout << "1/(sigmaˆ2*(K-1) + 0.5*r): " << (1/(pow(computeParam.volatility, 2)*(computeParam.K - 1) + 0.5*computeParam.risk_free_rate));
	std::cout << " delta: " << computeParam.delta_t << std::endl;

	if (computeParam.delta_t < (1/(pow(computeParam.volatility, 2)*(computeParam.N - 1) + 0.5*computeParam.risk_free_rate)))
	{
		return true;
	}
	else
	{
		return false;
	}
}
// function to compare difference of two grids
double square_error(float* current, float* next, struct GridParameter gridData)
{
	double sum = 0;

	for(unsigned int bat = 0; bat < gridData.batch; bat++)
	{
		int offset = bat * gridData.grid_size_x* gridData.grid_size_y;

		for(unsigned int i = 0; i < gridData.act_size_y; i++)
		{
			for(unsigned int j = 0; j < gridData.act_size_x; j++)
			{
				int index = i*gridData.grid_size_x + j+offset;
				float v1 = (next[index]);
				float v2 = (current[index]);

				if(fabs(v1-v2)/(fabs(v1) + fabs(v2)) >= 0.000001 && (fabs(v1) + fabs(v2)) > 0.000001 )
				{
					printf("i:%d j:%d v1:%f v2:%f\n", i, j, v1, v2);
				}

				sum += next[index]*next[index] - current[index]*current[index];
			}
		}
	}
    return sum;
}

int copy_grid(float* grid_s, float* grid_d, struct GridParameter gridData)
{
    for(unsigned int bat = 0; bat < gridData.batch; bat++)
    {
    	int offset = bat * gridData.grid_size_x * gridData.grid_size_y;

		for(unsigned int i = 0; i < gridData.act_size_y; i++)
		{
			for(unsigned int j = 0; j < gridData.act_size_x; j++)
			{
				grid_d[i*gridData.grid_size_x + j+offset] = grid_s[i*gridData.grid_size_x + j+offset];
			}
		}
    }
    return 0;
}

void intialize_grid(float* grid, GridParameter gridProp, BlacksholesParameter computeParam)
{
	float sMax = computeParam.strike_price * computeParam.SMaxFactor;

	for (unsigned int bat = 0; bat < gridProp.batch; bat++)
	{
		int offset = bat * gridProp.grid_size_x * gridProp.grid_size_y;

		for (unsigned int i = 0; i < gridProp.act_size_y; i++)
		{
			for (unsigned int j = 0; j < gridProp.act_size_x; j++)
			{
				if (j == 0)
				{
					grid[offset + i * gridProp.grid_size_x + j] = 0;
				}
				else if (j == gridProp.act_size_x -1)
				{
					grid[offset + i * gridProp.grid_size_x + j] = sMax;
				}
				else
				{
					grid[offset + i * gridProp.grid_size_x + j] = std::max(j*computeParam.delta_S - computeParam.strike_price, (float)0);
				}

//				std::cout << "grid_id: " << offset + i * gridData.grid_size_x + j << " val: " << grid[offset + i * gridData.grid_size_x + j] << std::endl;
			}
		}
	}
}

