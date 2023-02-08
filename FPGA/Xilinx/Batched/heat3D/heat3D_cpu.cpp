/** @author Beniel Thileepan
  * 
  */

#include "heat3D_cpu.h"

int heat3D_explicit(float * current, float *next, GridParameter gridData, std::vector<heat3DParameter> calcParam)
{
    assert(calcParam.size() == gridData.batch);

    for (unsigned int bat = 0; bat < gridData.batch; bat++)
    {
        //constant coefficients
        float coeff[] = {1 - 6 * calcParam[bat].K, calcParam[bat].K, calcParam[bat].K, 
                calcParam[bat].K, calcParam[bat].K, calcParam[bat].K, calcParam[bat].K};
        int offset = bat * gridData.grid_size_x * gridData.grid_size_y * gridData.grid_size_z; 
        float error = ERROR_TOL + 1; //intially setting the error greater than the tolarance
        unsigned int iter = 0;

        while (error > ERROR_TOL && iter < gridData.num_iter)
        {

            for (unsigned int k = 1; k < gridData.act_size_z - 1; k++)
            {
                for (unsigned int j = 1; j < gridData.act_size_y - 1; j++)
                {
                    for (unsigned int i = 1; i < gridData.act_size_x - 1; i++)
                    {
                        int index = offset + k * gridData.grid_size_x * gridData.grid_size_y 
                                + j * gridData.grid_size_x + i;
                        int index_i_min_1 = index - 1;
                        int index_i_pls_1 = index + 1;
                        int index_j_min_1 = index - gridData.grid_size_x;
                        int index_j_pls_1 = index + gridData.grid_size_x;
                        int index_k_min_1 = index - gridData.grid_size_x * gridData.grid_size_y;
                        int index_k_pls_1 = index + gridData.grid_size_x * gridData.grid_size_y;

                        next[index] = coeff[0] * current[index] 
                                + coeff[1] * current[index_i_min_1] + coeff[2] * current[index_i_pls_1]
                                + coeff[1] * current[index_j_min_1] + coeff[2] * current[index_j_pls_1]
                                + coeff[1] * current[index_k_min_1] + coeff[2] * current[index_k_pls_1];

                        error = fmax(error, fabs(next[index] - current[index]));
                    }

                    for (unsigned int i = 1; i < gridData.act_size_x - 1; i++)
                    {
                        int index = offset + k * gridData.grid_size_x * gridData.grid_size_y 
                                + j * gridData.grid_size_x + i;
                        int index_i_min_1 = index - 1;
                        int index_i_pls_1 = index + 1;
                        int index_j_min_1 = index - gridData.grid_size_x;
                        int index_j_pls_1 = index + gridData.grid_size_x;
                        int index_k_min_1 = index - gridData.grid_size_x * gridData.grid_size_y;
                        int index_k_pls_1 = index + gridData.grid_size_x * gridData.grid_size_y;

                        current[index] = coeff[0] * next[index] 
                                + coeff[1] * next[index_i_min_1] + coeff[2] * next[index_i_pls_1]
                                + coeff[1] * next[index_j_min_1] + coeff[2] * next[index_j_pls_1]
                                + coeff[1] * next[index_k_min_1] + coeff[2] * next[index_k_pls_1];
                        
                        error = fmax(error, fabs(next[index] - current[index]));
                    }
                }
            }
            iter += 2;
        }

        if (iter < gridData.num_iter)
        {
            std::cout << "Exiting iteration: " << iter << " with max error in iteration: " 
                    << error << " is less than tolarance: " << ERROR_TOL << std::endl; 
        }
        else
        {
            std::cout << "Exiting iteration: " << iter << " with max error in iteration: " 
                    << error << std::endl; 
        }
    }

    return 0;
}

void initialize_grid(float* grid, GridParameter gridData)
{
    float angle_res_x = 2 * M_PI / gridData.logical_size_x;
    float angle_res_y = 2 * M_PI / gridData.logical_size_y;
    float angle_res_z = 2 * M_PI / gridData.logical_size_z;

    for (unsigned int bat = 0; bat < gridData.batch; bat++)
    {
        int offset = bat * gridData.grid_size_x * gridData.grid_size_y * gridData.grid_size_z;

        for (unsigned int k = 0; k < gridData.act_size_z; k++)
        {
            for (unsigned int j = 0; j < gridData.act_size_y; j++)
            {
                for (unsigned int i = 0; i < gridData.act_size_x; i++)
                {
                    int index = offset + k * gridData.grid_size_x * gridData.grid_size_y 
                            + j * gridData.grid_size_x + i;

                    if (i == 0 or j == 0 or k == 0 or i == gridData.act_size_x - 1 
                            or j == gridData.act_size_y - 1 or k == gridData.act_size_z - 1)
                            {
                                grid[index] = 0;
                            }
                    else
                    {
                        grid[index] = sin(angle_res_x * (i-1)) 
                                * sin(angle_res_y * (j-1)) * sin(angle_res_z * (k-1));
                    }
                }
            }
        }
    }
}

int copy_grid(float* grid_s, float* grid_d, GridParameter gridData)
{
    for (unsigned int bat = 0; bat < gridData.batch; bat++)
    {
        int offset = bat * gridData.grid_size_x * gridData.grid_size_y * gridData.grid_size_z;

        for (unsigned int k = 0; k < gridData.act_size_z; k++)
        {
            for (unsigned int j = 0; j < gridData.act_size_y; j++)
            {
                 for (unsigned int  i = 0; i < gridData.act_size_x; i++)
                 {
                    int index = offset + k * gridData.grid_size_x * gridData.grid_size_y 
                            + j * gridData.grid_size_x + i;

                    grid_d[index] = grid_s[index];
                 }
            }
        }
    }
}
