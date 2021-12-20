//==============================================================
// Vector Add is the equivalent of a Hello, World! sample for data parallel
// programs. Building and running the sample verifies that your development
// environment is setup correctly and demonstrates the use of the core features
// of DPC++. This sample runs on both CPU and GPU (or FPGA). When run, it
// computes on both the CPU and offload device, then compares results. If the
// code executes on both CPU and offload device, the device name and a success
// message are displayed. And, your development environment is setup correctly!
//
// For comprehensive instructions regarding DPC++ Programming, go to
// https://software.intel.com/en-us/oneapi-programming-guide and search based on
// relevant terms noted in the comments.
//
// DPC++ material used in the code sample:
// •  A one dimensional array of data.
// •  A device queue, buffer, accessor, and kernel.
//==============================================================
// Copyright © Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <CL/sycl.hpp>
#include <vector>
#include <iostream>
#include "dpc_common.hpp"
#if FPGA || FPGA_EMULATOR
#include <CL/sycl/INTEL/fpga_extensions.hpp>
#include <CL/sycl/INTEL/ac_types/ac_int.hpp>
#endif

#include "rtm.h"
using namespace sycl;





const int unroll_factor = 2;
const int v_factor = 16;

struct dPath {
  [[intel::fpga_register]] float data[8];
};


struct dPath16 {
  [[intel::fpga_register]] float data[16];
};

// Vector type and data size for this example.
size_t vector_size = 10000;
typedef std::vector<struct dPath16> IntVector; 
typedef std::vector<float> IntVectorS; 

using rd_pipe = INTEL::pipe<class pVec16_r, dPath16, 8>;
using wr_pipe = INTEL::pipe<class pVec16_w, dPath16, 8>;

#define UFACTOR 2

struct pipeS{
  pipeS() = delete;
  template <size_t idx>  struct struct_id;

  template <size_t idx>
  struct Pipes{
    using pipeA = INTEL::pipe<struct_id<idx>, dPath, 8>;
  };

  template <size_t idx>
  using PipeAt = typename Pipes<idx>::pipeA;
};


using PipeBlock = pipeS;

// using rd_pipe1 = INTEL::pipe<class rd_pipe1, dPath, 8>;
// using wr_pipe1 = INTEL::pipe<class wr_pipe1, dPath, 8>;


// Create an exception handler for asynchronous SYCL exceptions
static auto exception_handler = [](sycl::exception_list e_list) {
  for (std::exception_ptr const &e : e_list) {
    try {
      std::rethrow_exception(e);
    }
    catch (std::exception const &e) {
#if _DEBUG
      std::cout << "Failure" << std::endl;
#endif
      std::terminate();
    }
  }
};

#include "populate_cpu.cpp"
#include "derives_calc_ytep_k1.cpp"
#include "derives_calc_ytep_k2.cpp"
#include "derives_calc_ytep_k3.cpp"
#include "derives_calc_ytep_k4.cpp"

template<int VFACTOR>
void stencil_read(queue &q, buffer<struct dPath16, 1> &in_buf, int total_itr){
      event e1 = q.submit([&](handler &h) {
      // cl::sycl::stream out(1024, 256, h);
      accessor in(in_buf, h, read_only);

      // int total_itr = ((nx*ny)*(nz))/VFACTOR;

      h.single_task<class producer>([=] () [[intel::kernel_args_restrict]]{

        [[intel::initiation_interval(1)]]
        for(int i = 0; i < total_itr; i++){
          struct dPath16 vec = in[i];
          rd_pipe::write(vec);
        }
        // out << "Finished reading the data\n";
        
      });
      });
}




template<int VFACTOR>
void PipeConvert_512_256(queue &q, int total_itr){
      event e1 = q.submit([&](handler &h) {

      // int total_itr = ((nx*ny)*(nz))/(VFACTOR);
      h.single_task<class PipeConvert_512_256>([=] () [[intel::kernel_args_restrict]]{
        struct dPath16 data16;
        [[intel::initiation_interval(1)]]
        for(int i = 0; i < total_itr; i++){
          struct dPath data;
          if((i&1) == 0){
            data16 = rd_pipe::read();
          }

          #pragma unroll VFACTOR
          for(int v = 0; v < VFACTOR; v++){
            if((i&1) == 0){
              data.data[v] = data16.data[v];
            } else {
              data.data[v] = data16.data[v+VFACTOR];
            }
          }
          pipeS::PipeAt<0>::write(data);
        }
        
      });
    });
}


template <int idx, int VFACTOR>
void PipeConvert_256_512(queue &q, int total_itr){
    event e3 = q.submit([&](handler &h) {
    // accessor out(out_buf, h, write_only);
    h.single_task<class pipeConvert_256_512>([=] () [[intel::kernel_args_restrict]]{
      // int total_itr = ((nx*ny)*(nz))/(VFACTOR);
      struct dPath16 data16;
      [[intel::initiation_interval(1)]]
      for(int i = 0; i < total_itr; i++){
        struct dPath data;
        data = pipeS::PipeAt<idx>::read();
        #pragma unroll VFACTOR
        for(int v = 0; v < VFACTOR; v++){
          if((i & 1) == 0){
            data16.data[v] = data.data[v];
          } else {
            data16.data[v+VFACTOR] = data.data[v];
          }
        }
        if((i & 1) == 1){
          wr_pipe::write(data16);
        }
      }

      
      
    });
    });
}

template <int VFACTOR>
void stencil_write(queue &q, buffer<struct dPath16, 1> &out_buf, int total_itr, double &kernel_time){
    event e3 = q.submit([&](handler &h) {
    accessor out(out_buf, h, write_only);
    std::string instance_name="consumer";
    h.single_task<class instance_name>([=] () [[intel::kernel_args_restrict]]{
      // int total_itr = ((nx*ny)*(nz))/VFACTOR;
      [[intel::initiation_interval(1)]]
      for(int i = 0; i < total_itr; i++){
        struct dPath16 vec = wr_pipe::read();
        out[i] = vec;
      }
      
    });
    });

    double start0 = e3.get_profiling_info<info::event_profiling::command_start>();
    double end0 = e3.get_profiling_info<info::event_profiling::command_end>(); 
    kernel_time += (end0-start0)*1e-9;
}


// template <int N, int n> struct loop {
//   static void instantiate(queue &q, int nx, int ny, int nz, int total_itr){
//     loop<N-1, n-1>::instantiate(q, nx, ny, nz, total_itr);
//     stencil_compute<N-1, n-1, 4096, 8>(q, nx, ny, nz, total_itr);
//   }
// };

// template<> 
// struct loop<1, 1>{
//   static void instantiate(queue &q, int nx, int ny, int nz, int total_itr){
//     stencil_compute<0, 0, 4096, 8>(q, nx, ny, nz, total_itr);
//   }
// };

// loop<90> l;


//************************************
// Vector add in DPC++ on device: returns sum in 4th parameter "sum_parallel".
//************************************
void stencil_comp(queue &q, IntVector &input, IntVector &output, int n_iter, int nx, int ny, int nz, int batch) {
  // Create the range object for the vectors managed by the buffer.
  range<1> num_items{input.size()};
  int vec_size = input.size();

  // Create buffers that hold the data shared between the host and the devices.
  // The buffer destructor is responsible to copy the data back to host when it
  // goes out of scope.
  buffer in_buf(input);
  buffer out_buf(output);

  // Submit a command group to the queue by a lambda function that contains the
  // data access permission and device computation (kernel).
  double kernel_time = 0;
  std::cout << "starting writing to the pipe\n" << std::endl;
  dpc_common::TimeInterval exe_time;


  struct data_G data_g;
  data_g.sizex = nx;
  data_g.sizey = ny;
  data_g.sizez = nz;
  data_g.grid_sizex = nx+2*ORDER;
  data_g.grid_sizey = ny+2*ORDER;
  data_g.grid_sizez = nz+2*ORDER;
  data_g.limit_z = nz+3*ORDER;


  unsigned short grid_sizey_4 = (data_g.grid_sizey - 4);
  data_g.plane_size = data_g.grid_sizex * data_g.grid_sizey;

  data_g.plane_diff = data_g.grid_sizex * grid_sizey_4;
  data_g.line_diff = data_g.grid_sizex - 4;
  data_g.gridsize_pr = data_g.plane_size * (data_g.limit_z) * batch;

  unsigned int total_itr_8 = (data_g.plane_size * (data_g.grid_sizez)* batch);
  unsigned int total_itr_16 = total_itr_8 >> 1;

    for(int itr = 0; itr < n_iter; itr++){

    // reading from memory
      stencil_read<16>(q, in_buf, total_itr_16);
      PipeConvert_512_256<8>(q, total_itr_8);

      derives_calc_ytep_k1( q, data_g);
      derives_calc_ytep_k2( q, data_g);
      derives_calc_ytep_k3( q, data_g);
      derives_calc_ytep_k4( q, data_g);

      PipeConvert_256_512<4, 8>(q, total_itr_8);
      //write back to memory
      stencil_write<16>(q, out_buf, total_itr_16, kernel_time);
      q.wait();

      
      // // // reading from memory
      stencil_read<16>(q, out_buf, total_itr_16);
      PipeConvert_512_256<8>(q, total_itr_8);

      derives_calc_ytep_k1( q, data_g);
      derives_calc_ytep_k2( q, data_g);
      derives_calc_ytep_k3( q, data_g);
      derives_calc_ytep_k4( q, data_g);

      PipeConvert_256_512<4, 8>(q, total_itr_8);
      //write back to memory
      stencil_write<16>(q, in_buf, total_itr_16, kernel_time);
      q.wait();
    }

    std::cout << "fimished reading from the pipe\n" << std::endl;

    double exe_elapsed = exe_time.Elapsed();
    double bandwidth = 2.0*v_factor*vec_size*sizeof(int)*n_iter*2.0/(kernel_time*1000000000);
    std::cout << "Elapsed time: " << kernel_time << std::endl;
    std::cout << "Bandwidth(GB/s): " << bandwidth << std::endl;
}

//************************************
// Initialize the vector from 0 to vector_size - 1
//************************************
template<int VFACTOR>
void InitializeVector(IntVector &a) {
  for (size_t i = 0; i < a.size(); i++){
    for(int v = 0; v < VFACTOR; v++){
        a[i].data[v] = i* VFACTOR + v +0.5f;
    }
  }
}

void InitializeVectorS(IntVectorS &a) {
  for (size_t i = 0; i < a.size(); i++){
      a[i] = i+0.5f;
  }
}

//************************************
// Demonstrate vector add both in sequential on CPU and in parallel on device.
//************************************
int main(int argc, char* argv[]) {

  int n_iter = 1;
  int nx = 128, ny = 128, nz=2, batch=1;
  // Change vector_size if it was passed as argument
  if (argc > 1) n_iter = std::stoi(argv[1]);
  if (argc > 2) nx = std::stoi(argv[2]);
  if (argc > 3) ny = std::stoi(argv[3]);
  if (argc > 4) nz = std::stoi(argv[4]);
  if (argc > 5) batch = std::stoi(argv[5]);

  nx = (nx % 8 == 0 ? nx : (nx/8+1)*8);

  struct Grid_d grid_d;
  grid_d.logical_size_x = nx;
  grid_d.logical_size_y = ny;
  grid_d.logical_size_z = nz;


  grid_d.act_sizex = nx + ORDER*2;
  grid_d.act_sizey = ny + ORDER*2;
  grid_d.act_sizez = nz + ORDER*2;

  grid_d.grid_size_x = grid_d.act_sizex;
  grid_d.grid_size_y = grid_d.act_sizey;
  grid_d.grid_size_z = grid_d.act_sizez;


  grid_d.data_size_bytes_dim1 = 1*grid_d.grid_size_x * grid_d.grid_size_y * grid_d.grid_size_z* sizeof(float) * batch;
  grid_d.data_size_bytes_dim6 = 6*grid_d.grid_size_x * grid_d.grid_size_y * grid_d.grid_size_z* sizeof(float) * batch;
  grid_d.data_size_bytes_dim8 = 8*grid_d.grid_size_x * grid_d.grid_size_y * grid_d.grid_size_z* sizeof(float) * batch;
  grid_d.dims = 8*grid_d.grid_size_x * grid_d.grid_size_y * grid_d.grid_size_z;


  float * grid_yy_rho_mu              = (float*)aligned_alloc(4096, grid_d.data_size_bytes_dim8);
  float * grid_yy_rho_mu_temp         = (float*)aligned_alloc(4096, grid_d.data_size_bytes_dim8);
  float * grid_k1                     = (float*)aligned_alloc(4096, grid_d.data_size_bytes_dim8);
  float * grid_k2                     = (float*)aligned_alloc(4096, grid_d.data_size_bytes_dim8);
  float * grid_k3                     = (float*)aligned_alloc(4096, grid_d.data_size_bytes_dim8);
  float * grid_k4                     = (float*)aligned_alloc(4096, grid_d.data_size_bytes_dim8);

  // need to be vector, will change that later
  // float * grid_yy_rho_mu_d     = (float*)aligned_alloc(4096, grid_d.data_size_bytes_dim8);
  // float * grid_yy_rho_mu_temp_d  = (float*)aligned_alloc(4096, grid_d.data_size_bytes_dim8);

  IntVector grid_yy_rho_mu_d, grid_yy_rho_mu_temp_d;
  grid_yy_rho_mu_d.resize(grid_d.data_size_bytes_dim8/(v_factor*sizeof(float)));
  grid_yy_rho_mu_temp_d.resize(grid_d.data_size_bytes_dim8/(v_factor*sizeof(float)));

  // printf("grid_d.data_size_bytes_dim8:%d\n", grid_d.data_size_bytes_dim8);

  for(int i = 0; i < batch; i++){
    populate_rho_mu_yy(&grid_yy_rho_mu[grid_d.dims * i], grid_d);
  }
  copy_grid(grid_yy_rho_mu, grid_yy_rho_mu_d, grid_d.data_size_bytes_dim8);


  float dt = 0.1;
  for(int i = 0; i < batch; i++){
    for(int itr = 0; itr < n_iter*2; itr++){
       fd3d_pml_kernel(&grid_yy_rho_mu[grid_d.dims * i], &grid_k1[grid_d.dims * i], grid_d);
       calc_ytemp_kernel(&grid_yy_rho_mu[grid_d.dims * i], &grid_k1[grid_d.dims * i], dt, &grid_yy_rho_mu_temp[grid_d.dims * i], 0.5, grid_d);

       fd3d_pml_kernel(&grid_yy_rho_mu_temp[grid_d.dims * i], &grid_k2[grid_d.dims * i], grid_d);
       calc_ytemp_kernel(&grid_yy_rho_mu[grid_d.dims * i], &grid_k2[grid_d.dims * i], dt, &grid_yy_rho_mu_temp[grid_d.dims * i], 0.5, grid_d);

       fd3d_pml_kernel(&grid_yy_rho_mu_temp[grid_d.dims * i], &grid_k3[grid_d.dims * i], grid_d);
       calc_ytemp_kernel(&grid_yy_rho_mu[grid_d.dims * i], &grid_k3[grid_d.dims * i], dt, &grid_yy_rho_mu_temp[grid_d.dims * i], 1.0, grid_d);

       fd3d_pml_kernel(&grid_yy_rho_mu_temp[grid_d.dims * i], &grid_k4[grid_d.dims * i], grid_d);
       final_update_kernel(&grid_yy_rho_mu[grid_d.dims * i], &grid_k1[grid_d.dims * i], &grid_k2[grid_d.dims * i], &grid_k3[grid_d.dims * i], &grid_k4[grid_d.dims * i], dt, grid_d);

     }
   }


  // Create device selector for the device of your interest.
#if FPGA_EMULATOR
  // DPC++ extension: FPGA emulator selector on systems without FPGA card.
  INTEL::fpga_emulator_selector d_selector;
#elif FPGA
  // DPC++ extension: FPGA selector on systems with FPGA card.
  INTEL::fpga_selector d_selector;
#else
  // The default device selector will select the most performant device.
  default_selector d_selector;
#endif


  try {
    queue q(d_selector,  dpc_common::exception_handler, property::queue::enable_profiling{});

    // queue q2(d_selector,  dpc_common::exception_handler);


    // Print out the device information used for the kernel code.
    std::cout << "Running on device: "
              << q.get_device().get_info<info::device::name>() << "\n";

    // Vector addition in DPC++
    
    stencil_comp(q, grid_yy_rho_mu_d, grid_yy_rho_mu_temp_d, n_iter, nx, ny, nz, batch);

  } catch (exception const &e) {
    std::cout << "An exception is caught for vector add.\n";
    std::terminate();
  }



  // Compute the sum of two vectors in sequential for validation.

   std::cout << "No error until here\n";

   const int struct_s = 8;

  // Verify that the two vectors are equal. 
  int xblk = (nx+2*ORDER);
  for(int k = 0; k < grid_d.grid_size_z; k++){ 
    for(int j = 0; j < grid_d.grid_size_y; j++){
      for(int i = 0; i < grid_d.grid_size_x; i++){
        // std::cout << "i, j, k, grid_yy_rho_mu[ind+v]: " << i << " " << j << " " << k << " "; 
        for(int v = 0; v < struct_s; v++){
          int ind = (k*grid_d.grid_size_x*grid_d.grid_size_y + j*grid_d.grid_size_x + i)*struct_s;
          int s_ind = v+ (i & 1)*8;
          float chk = fabs((grid_yy_rho_mu[ind+v] - grid_yy_rho_mu_d.at(ind/v_factor).data[s_ind])/(grid_yy_rho_mu[ind+v]));
          
          // std::cout << "(" << grid_yy_rho_mu[ind+v] << "," << grid_yy_rho_mu_d.at(ind/v_factor).data[s_ind] << ") ";
          if(chk > 0.0001 && fabs(grid_yy_rho_mu[ind+v]) > 0.00001 && !isnan(chk)){
            std::cout << "j,i, k, ind: " << j  << " " << i*v_factor+v << " " << k << " " << ind << " " << grid_yy_rho_mu[ind+v] << " " << grid_yy_rho_mu_d.at(ind/v_factor).data[s_ind] <<  std::endl;
            return -1;
          }
        }
        // std::cout << "\n";
      }
    }
  }


  free(grid_yy_rho_mu);
  free(grid_yy_rho_mu_temp);
  free(grid_k1);
  free(grid_k2);
  free(grid_k3);
  free(grid_k4);

  grid_yy_rho_mu_d.clear();
  grid_yy_rho_mu_temp_d.clear();


  std::cout << "Vector add successfully completed on device.\n";
  return 0;
}
