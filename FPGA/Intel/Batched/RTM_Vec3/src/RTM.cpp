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
const int v_factor = 24;

struct dPath {
  [[intel::fpga_register]]
   float data[24];
};


struct dPath18 {
  [[intel::fpga_register]]
   float data[18];
};


struct dPath16 {
  [[intel::fpga_register]]
   float data[16];
};

struct dPath2 {
  [[intel::fpga_register]]
   float data[2];
};

// Vector type and data size for this example.
size_t vector_size = 10000;
typedef std::vector<struct dPath16> IntVector; 
typedef std::vector<float> IntVectorS; 


#define UFACTOR 2

struct pipeS{
  pipeS() = delete;
  template <size_t idx>  struct struct_idS;

  template <size_t idx>
  struct Pipes1{
    using pipeA = INTEL::pipe<struct_idS<idx>, dPath16, 1024>;
  };

  template <size_t idx>
  using PipeAt = typename Pipes1<idx>::pipeA;
};

struct pipeM{
  pipeM() = delete;
  template <size_t idx>  struct struct_idM;

  template <size_t idx>
  struct Pipes2{
    using pipeB = INTEL::pipe<struct_idM<idx>, dPath, 8>;
  };

  template <size_t idx>
  using PipeAt = typename Pipes2<idx>::pipeB;
};


// using PipeBlock = pipeS;

template <size_t idx>  struct struct_idX;

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

template<int idx1, int idx2, int VFACTOR>
event stencil_read_write(queue &q, buffer<struct dPath16, 1> &in_buf1, buffer<struct dPath16, 1> &in_buf2, buffer<struct dPath16, 1> &in_buf3, 
  buffer<struct dPath16, 1> &out_buf1, buffer<struct dPath16, 1> &out_buf2, buffer<struct dPath16, 1> &out_buf3, 
  int total_itr, ac_int<12,true> n_iter, int delay
  ){
      event e1 = q.submit([&](handler &h) {

      accessor in1(in_buf1, h);
      accessor in2(in_buf2, h);
      accessor in3(in_buf3, h);

      accessor out1(out_buf1, h);
      accessor out2(out_buf2, h);
      accessor out3(out_buf3, h);

      h.single_task<class stencil_read>([=] () [[intel::kernel_args_restrict]]{

      [[intel::disable_loop_pipelining]]
      for(ac_int<12,true> itr = 0; itr < n_iter; itr++){

        accessor ptrR1 = ((itr & 1) == 0) ? in1 : out1;
        accessor ptrR2 = ((itr & 1) == 0) ? in2 : out2;
        accessor ptrR3 = ((itr & 1) == 0) ? in3 : out3;


        accessor ptrW1 = ((itr & 1) == 1) ? in1 : out1;
        accessor ptrW2 = ((itr & 1) == 1) ? in2 : out2;
        accessor ptrW3 = ((itr & 1) == 1) ? in3 : out3;

        [[intel::ivdep]]
        [[intel::initiation_interval(1)]]
        for(int i = 0; i < total_itr+delay; i++){

          struct dPath16 vecR1 = ptrR1[i+delay];
          struct dPath16 vecR2 = ptrR2[i+delay];
          struct dPath16 vecR3 = ptrR3[i+delay];

          if(i < total_itr){
            pipeS::PipeAt<idx1>::write(vecR1);
            pipeS::PipeAt<idx1+1>::write(vecR2);
            pipeS::PipeAt<idx1+2>::write(vecR3);
          }

          struct dPath16 vecW1;
          struct dPath16 vecW2;
          struct dPath16 vecW3;


          if(i >= delay){

            vecW1 = pipeS::PipeAt<idx2>::read();
            vecW2 = pipeS::PipeAt<idx2+1>::read();
            vecW3 = pipeS::PipeAt<idx2+2>::read();

          }

          ptrW1[i] = vecW1;
          ptrW2[i] = vecW2;
          ptrW3[i] = vecW3;

        }
      }        
      });
      });

      return e1;
}




template<int idx1, int idx2, int VFACTOR>
void PipeConvert_3_1(queue &q, int total_itr, ac_int<12,true> n_iter){

      ac_int<40,true> count = total_itr * n_iter;
      event e1 = q.submit([&](handler &h) {

      h.single_task<class PipeConvert_512_256>([=] () [[intel::kernel_args_restrict]]{
        struct dPath16 data1, data2, data3;
        [[intel::initiation_interval(1)]]
        for(int i = 0; i < count; i++){
          
          struct dPath out;
          if((i & 1) == 0){
            data1 = pipeS::PipeAt<idx1>::read();
            data2 = pipeS::PipeAt<idx1+1>::read();
            data3 = pipeS::PipeAt<idx1+2>::read();
          } 

          struct dPath out1 = {{data1.data[0], data1.data[1], data1.data[2], data1.data[3], data1.data[4], data1.data[5], data1.data[6], data1.data[7],
                              data1.data[8], data1.data[9], data1.data[10], data1.data[11], data1.data[12], data1.data[13], data1.data[14], data1.data[15],
                              data2.data[0], data2.data[1], data2.data[2], data2.data[3], data2.data[4], data2.data[5], data2.data[6], data2.data[7]

           }};

          struct dPath out2 = {{data2.data[8], data2.data[9], data2.data[10], data2.data[11], data2.data[12], data2.data[13], data2.data[14], data2.data[15],
                              data3.data[0], data3.data[1], data3.data[2], data3.data[3], data3.data[4], data3.data[5], data3.data[6], data3.data[7],
                              data3.data[8], data3.data[9], data3.data[10], data3.data[11], data3.data[12], data3.data[13], data3.data[14], data3.data[15],
           }};

          if((i & 1) == 0){
            out = out1;
          } else {
            out = out2;
          }
          pipeM::PipeAt<idx2>::write(out);
        }
        
      });
    });
}


template <int idx1, int idx2, int VFACTOR>
void PipeConvert_1_3(queue &q, int total_itr,  ac_int<12,true> n_iter){
    ac_int<40,true> count = total_itr * n_iter;
    event e3 = q.submit([&](handler &h) {
    h.single_task<class pipeConvert_256_512>([=] () [[intel::kernel_args_restrict]]{
      struct dPath out1, out2, out;
      [[intel::initiation_interval(1)]]
      for(int i = 0; i < count; i++){
          
          out = pipeM::PipeAt<idx1>::read();

          if((i&1) == 0) {
            out1 = out;
          } else {
            out2 = out;
          }
          
          struct dPath16 data1 = {{out1.data[0], out1.data[1], out1.data[2], out1.data[3], out1.data[4], out1.data[5], out1.data[6], out1.data[7],
                                  out1.data[8], out1.data[9], out1.data[10], out1.data[11], out1.data[12], out1.data[13], out1.data[14], out1.data[15]
          }};

          struct dPath16 data2 = {{out1.data[16], out1.data[17], out1.data[18], out1.data[19], out1.data[20], out1.data[21], out1.data[22], out1.data[23],
                                  out2.data[0], out2.data[1], out2.data[2], out2.data[3], out2.data[4], out2.data[5], out2.data[6], out2.data[7]
          }};

          struct dPath16 data3 = {{out2.data[8], out2.data[9], out2.data[10], out2.data[11], out2.data[12], out2.data[13], out2.data[14], out2.data[15],
                                  out2.data[16], out2.data[17], out2.data[18], out2.data[19], out2.data[20], out2.data[21], out2.data[22], out2.data[23]
          }};


          if((i&1) == 1){
            pipeS::PipeAt<idx2>::write(data1);
            pipeS::PipeAt<idx2+1>::write(data2);
            pipeS::PipeAt<idx2+2>::write(data3);
          }

      }

      
      
    });
    });
}

// template <int idx, int VFACTOR>
// void stencil_write(queue &q, buffer<struct dPath16, 1> &out_buf1, buffer<struct dPath16, 1> &out_buf2, buffer<struct dPath16, 1> &out_buf3, 
//   int total_itr, double &kernel_time){
//     event e3 = q.submit([&](handler &h) {
//     accessor out1(out_buf1, h, write_only);
//     accessor out2(out_buf2, h, write_only);
//     accessor out3(out_buf3, h, write_only);
//     std::string instance_name="consumer";
//     h.single_task<class stencil_write>([=] () [[intel::kernel_args_restrict]]{
//       // int total_itr = ((nx*ny)*(nz))/VFACTOR;
//       [[intel::initiation_interval(1)]]
//       for(int i = 0; i < total_itr; i++){
//         struct dPath16 vec1 = pipeS::PipeAt<idx>::read();
//         struct dPath16 vec2 = pipeS::PipeAt<idx+1>::read();
//         struct dPath16 vec3 = pipeS::PipeAt<idx+2>::read();
//         out1[i] = vec1;
//         out2[i] = vec2;
//         out3[i] = vec3;
//       }
      
//     });
//     });

//     double start0 = e3.get_profiling_info<info::event_profiling::command_start>();
//     double end0 = e3.get_profiling_info<info::event_profiling::command_end>(); 
//     kernel_time += (end0-start0)*1e-9;
// }


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
void stencil_comp(queue &q, IntVector &input1, IntVector &output1, IntVector &input2, IntVector &output2,
   IntVector &input3, IntVector &output3,
   int n_iter, int nx, int ny, int nz, int batch, int delay) {
  // Create the range object for the vectors managed by the buffer.
  range<1> num_items{input1.size()};
  int vec_size = input1.size()*3;

  // Create buffers that hold the data shared between the host and the devices.
  // The buffer destructor is responsible to copy the data back to host when it
  // goes out of scope.
  buffer in_buf1(input1);
  buffer out_buf1(output1);


  buffer in_buf2(input2);
  buffer out_buf2(output2);

  buffer in_buf3(input3);
  buffer out_buf3(output3);

  // Submit a command group to the queue by a lambda function that contains the
  // data access permission and device computation (kernel).
  double kernel_time = 0, kernel_time1 = 0;
  std::cout << "starting writing to the pipe\n" << std::endl;
  dpc_common::TimeInterval exe_time;


  struct data_G data_g;
  data_g.sizex = nx;
  data_g.sizey = ny;
  data_g.sizez = nz;
  data_g.grid_sizex = nx+2*ORDER;
  data_g.grid_sizey = ny+2*ORDER;
  data_g.grid_sizez = nz+2*ORDER;
  data_g.xblocks = data_g.grid_sizex/3;
  data_g.limit_z = nz+3*ORDER;


  unsigned short grid_sizey_4 = (data_g.grid_sizey - 4);
  data_g.plane_size = data_g.xblocks * data_g.grid_sizey;

  data_g.plane_diff = data_g.xblocks * grid_sizey_4;
  data_g.line_diff = data_g.xblocks - 2;
  data_g.gridsize_pr = data_g.plane_size * (data_g.grid_sizez * batch+ ORDER) ;
  data_g.rd_limit = data_g.plane_size*data_g.grid_sizez*batch;

  unsigned totol_8 = (data_g.plane_size * (data_g.grid_sizez)* batch);
  unsigned int total_itr_48 = totol_8/2;
  unsigned int total_itr_24 = totol_8;

    // for(int itr = 0; itr < n_iter; itr++){

    // reading from memory
      event e = stencil_read_write<0,3,16>(q, in_buf1, in_buf2, in_buf3, out_buf1, out_buf2, out_buf3, total_itr_48, n_iter, delay);
      PipeConvert_3_1<0,0, 8>(q, total_itr_24, n_iter);

      derives_calc_ytep_k1<0>( q, data_g, n_iter);
      derives_calc_ytep_k2<1>( q, data_g, n_iter);
      derives_calc_ytep_k3<2>( q, data_g, n_iter);
      derives_calc_ytep_k4<3>( q, data_g, n_iter);


      derives_calc_ytep_k1<4>( q, data_g, n_iter);
      derives_calc_ytep_k2<5>( q, data_g, n_iter);
      derives_calc_ytep_k3<6>( q, data_g, n_iter);
      derives_calc_ytep_k4<7>( q, data_g, n_iter);

      PipeConvert_1_3<8, 3, 8>(q, total_itr_24, n_iter);
      // stencil_write<3,16>(q, out_buf1, out_buf2, out_buf3, total_itr_48, kernel_time);
      q.wait();


      double start0 = e.get_profiling_info<info::event_profiling::command_start>();
      double end0 = e.get_profiling_info<info::event_profiling::command_end>(); 
      kernel_time += (end0-start0)*1e-9;

      
      // // // // // reading from memory
      // stencil_read<0,16>(q, out_buf1, out_buf2, out_buf3, total_itr_48);
      // // stencil_read<1,16>(q, in_buf2, total_itr_32);
      // PipeConvert_3_1<0,0, 8>(q, total_itr_24);

      // derives_calc_ytep_k1<0>( q, data_g);
      // derives_calc_ytep_k2<1>( q, data_g);
      // derives_calc_ytep_k3<2>( q, data_g);
      // derives_calc_ytep_k4<3>( q, data_g);

      // derives_calc_ytep_k1<4>( q, data_g);
      // derives_calc_ytep_k2<5>( q, data_g);
      // derives_calc_ytep_k3<6>( q, data_g);
      // derives_calc_ytep_k4<7>( q, data_g);

      // PipeConvert_1_3<8,3, 8>(q, total_itr_24);
      // stencil_write<3,16>(q, in_buf1, in_buf2, in_buf3, total_itr_48, kernel_time);

      // q.wait();
    // }

    std::cout << "fimished reading from the pipe\n" << std::endl;

    double exe_elapsed = exe_time.Elapsed();
    double bandwidth = 2.0* 3.0*v_factor*vec_size*sizeof(int)*n_iter/(kernel_time*1000000000);
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

  nx = ((nx+ORDER*2) % 6 == 0 ? nx : ((nx+ORDER*2)/6+1)*6 - ORDER*2);

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
  float * temp                        = (float*)aligned_alloc(4096, grid_d.data_size_bytes_dim8);

  // need to be vector, will change that later
  // float * grid_yy_rho_mu_d     = (float*)aligned_alloc(4096, grid_d.data_size_bytes_dim8);
  // float * grid_yy_rho_mu_temp_d  = (float*)aligned_alloc(4096, grid_d.data_size_bytes_dim8);

  IntVector grid_yy_rho_mu_d1, grid_yy_rho_mu_temp_d1;
  IntVector grid_yy_rho_mu_d2, grid_yy_rho_mu_temp_d2;
  IntVector grid_yy_rho_mu_d3, grid_yy_rho_mu_temp_d3;

  const int DDR_width = 16;
  int delay = 8*4*grid_d.grid_size_x/(6)*grid_d.grid_size_y+ 900 + 139*8/2+12/2; //872

  grid_yy_rho_mu_d1.resize(grid_d.data_size_bytes_dim8/(DDR_width*sizeof(float)*3) + delay*2);
  grid_yy_rho_mu_temp_d1.resize(grid_d.data_size_bytes_dim8/(DDR_width*sizeof(float)*3) + delay*2);
  grid_yy_rho_mu_d2.resize(grid_d.data_size_bytes_dim8/(DDR_width*sizeof(float)*3) + delay*2);
  grid_yy_rho_mu_temp_d2.resize(grid_d.data_size_bytes_dim8/(DDR_width*sizeof(float)*3) + delay*2);
  grid_yy_rho_mu_d3.resize(grid_d.data_size_bytes_dim8/(DDR_width*sizeof(float)*3) + delay*2);
  grid_yy_rho_mu_temp_d3.resize(grid_d.data_size_bytes_dim8/(DDR_width*sizeof(float)*3) + delay*2);


  // printf("grid_d.data_size_bytes_dim8:%d\n", grid_d.data_size_bytes_dim8);

  for(int i = 0; i < batch; i++){
    populate_rho_mu_yy(&grid_yy_rho_mu[grid_d.dims * i], grid_d);
  }
  copy_ToVec(grid_yy_rho_mu, grid_yy_rho_mu_d1, grid_yy_rho_mu_d2, grid_yy_rho_mu_d3, grid_d.data_size_bytes_dim8, delay);


  float dt = 0.1;



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
    
    stencil_comp(q, grid_yy_rho_mu_d1, grid_yy_rho_mu_temp_d1,  grid_yy_rho_mu_d2, grid_yy_rho_mu_temp_d2,  grid_yy_rho_mu_d3, grid_yy_rho_mu_temp_d3,
     n_iter*2, nx, ny, nz, batch, delay);

  } catch (exception const &e) {
    std::cout << "An exception is caught for vector add.\n";
    std::terminate();
  }


    for(int i = 0; i < batch; i++){
    for(int itr = 0; itr < 2*n_iter; itr++){
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

  copy_FromVec(grid_yy_rho_mu_d1, grid_yy_rho_mu_d2, grid_yy_rho_mu_d3, temp,  grid_d.data_size_bytes_dim8, delay);
  // for(int i = 0; i < grid_d.data_size_bytes_dim8/4; i++){
  //   float chk = fabs((grid_yy_rho_mu[i] - temp[i])/(grid_yy_rho_mu[i]));
  //   // std::cout << "i: " << i << " " << grid_yy_rho_mu_temp[i] << " " << temp[i] << " ";
  //   if(chk > 0.000001 && fabs(grid_yy_rho_mu[i]) > 0.000001 && !isnan(chk)){
  //     std::cout << "i: " << i << " " << grid_yy_rho_mu[i] << " " << temp[i] << "\n";
  //     // return -1;
  //   }
  // }


  for(int i = 0; i < grid_d.grid_size_z; i++){
    for(int j = 0; j < grid_d.grid_size_y; j++){
      for(int k = 0; k  < grid_d.grid_size_x; k++){
        for(int v= 0; v < 8; v++){
          int ind = (i*grid_d.grid_size_y*grid_d.grid_size_x + j*grid_d.grid_size_x+k)*8+v;
          float chk = fabs((grid_yy_rho_mu[ind] - temp[ind])/(grid_yy_rho_mu[ind]));
          // std::cout << "i: " << i << " " << grid_yy_rho_mu_temp[i] << " " << temp[i] << " ";
          if(chk > 0.0001 && fabs(grid_yy_rho_mu[ind]) > 0.0001 && !isnan(chk)){
            std::cout << "i,j,k,v, chk: " << i << " " << j << " " << k << " " << v << " " << chk << " " << grid_yy_rho_mu[ind] << " " << temp[ind] << "\n";
          }
        }
      }
    }
  }



  // Compute the sum of two vectors in sequential for validation.

   std::cout << "No error until here\n";



  free(grid_yy_rho_mu);
  free(grid_yy_rho_mu_temp);
  free(grid_k1);
  free(grid_k2);
  free(grid_k3);
  free(grid_k4);
  free(temp);

  grid_yy_rho_mu_d1.clear();
  grid_yy_rho_mu_temp_d1.clear();
  grid_yy_rho_mu_d2.clear();
  grid_yy_rho_mu_temp_d2.clear();
  grid_yy_rho_mu_d3.clear();
  grid_yy_rho_mu_temp_d3.clear();


  std::cout << "Vector add successfully completed on device.\n";
  return 0;
}
