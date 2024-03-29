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

using namespace sycl;

// Vector type and data size for this example.
size_t vector_size = 10000;
typedef std::vector<float> IntVector; 
const int unroll_factor = 2;

struct dPath {
  [[intel::fpga_register]] float data[8];
};

struct dPath16 {
  [[intel::fpga_register]] float data[16];
};

using rd_pipe = INTEL::pipe<class pVec16_r, dPath16, 8000000>;
using wr_pipe = INTEL::pipe<class pVec16_w, dPath16, 8000000>;

#define UFACTOR 2

struct pipeS{
  pipeS() = delete;
  template <size_t idx>  struct struct_id;

  template <size_t idx>
  struct Pipes{
    using pipeA = INTEL::pipe<struct_id<idx>, dPath, 8000000>;
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


template<int VFACTOR>
void stencil_read(queue &q, const float* in, ac_int<12,true> nx, ac_int<12,true> ny, ac_int<12,true> nz, ac_int<12,true> batch){
      event e1 = q.submit([&](handler &h) {

      int total_itr = ((nx*ny)*(nz*batch))/(VFACTOR*2);
      const int VFACTOR2 = VFACTOR*2;
      h.single_task<class producer>([=] () [[intel::kernel_args_restrict]]{

        [[intel::initiation_interval(1)]]
        for(int i = 0; i < total_itr; i++){
          struct dPath16 vec;
          #pragma unroll VFACTOR2
          for(int v = 0; v < VFACTOR2; v++){
            vec.data[v] = in[i*VFACTOR2+v];
          }
          rd_pipe::write(vec);
        }
        
      });
    });
}

template<int VFACTOR>
void PipeConvert_512_256(queue &q, ac_int<12,true> nx, ac_int<12,true> ny, ac_int<12,true> nz, ac_int<12,true> batch){
      event e1 = q.submit([&](handler &h) {

      int total_itr = ((nx*ny)*(nz*batch))/(VFACTOR);
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

template <size_t idx>  struct struct_idX;
template<int idx, int IdX, int DMAX, int VFACTOR>
void stencil_compute(queue &q, ac_int<14,true>  nx, ac_int<14,true>  ny, ac_int<14,true>  nz){
    event e2 = q.submit([&](handler &h) {
    std::string instance_name="compute"+std::to_string(idx);
    h.single_task<class struct_idX<IdX>>([=] () [[intel::kernel_args_restrict]]{
    
    int total_itr = ((nx/VFACTOR)*(ny*nz+1));
    struct dPath s_1_2, s_2_1, s_1_1, s_0_1, s_1_0;

    const int max_dpethl = DMAX/VFACTOR;

    struct dPath wind1[max_dpethl];
    struct dPath wind2[max_dpethl];

    struct dPath vec_wr;
    [[intel::fpga_register]] float mid_row[10];
    ac_int<14,true>  i_ld = 0;


    short id = 0, jd = 0, kd = 0;
    unsigned short rEnd = (nx/VFACTOR)-1;
    [[intel::initiation_interval(1)]]
    for(int itr = 0; itr < total_itr; itr++){
      ac_int<14,true>  i = id; 
      ac_int<14,true>  j = jd; 
      ac_int<14,true>  k = kd;
      ac_int<14,true>  i_l = i_ld;

      if(i == rEnd){
        id = 0;
      } else {
        id++;
      }

      if(i == rEnd && j == ny){
        jd = 1;
      } else if(i == rEnd){
        jd++;
      }

      if(i == rEnd && j == ny){
        kd++;
      }



      s_1_0 = wind2[i_l];

      s_0_1 = s_1_1;
      wind2[i_l] = s_0_1;

      s_1_1 = s_2_1;
      s_2_1 = wind1[i_l];

      if(itr < (nx/VFACTOR)*ny*nz){
        s_1_2 = pipeS::PipeAt<idx>::read();
      }

      wind1[i_l] = s_1_2;

      if(i_l >= nx/VFACTOR -2){
        i_ld = 0;
      } else {
        i_ld++;
      }

      #pragma unroll VFACTOR
      for(int v = 0; v < VFACTOR; v++){
        mid_row[v+1] = s_1_1.data[v]; 
      }
      mid_row[0] = s_0_1.data[VFACTOR-1];
      mid_row[VFACTOR+1] = s_2_1.data[0];

      #pragma unroll VFACTOR
      for(int v = 0; v < VFACTOR; v++){
        int i_ind = i *VFACTOR + v;
        float val =  (mid_row[v] + mid_row[v+2] + s_1_0.data[v] + s_1_2.data[v])*0.125f + (mid_row[v+1])*0.5f;
        vec_wr.data[v] = (i_ind > 0 && i_ind < nx-1 && j > 1 && j < ny ) ? val : mid_row[v+1];
      }
      if(itr >= (nx/VFACTOR)){
        pipeS::PipeAt<idx+1>::write(vec_wr);
      }
    }
    
  });
  });
}



template <int idx, int VFACTOR>
void PipeConvert_256_512(queue &q, ac_int<12,true> nx, ac_int<12,true> ny, ac_int<12,true> nz,
                  ac_int<12,true> batch){
    event e3 = q.submit([&](handler &h) {
    // accessor out(out_buf, h, write_only);
    h.single_task<class pipeConvert_256_512>([=] () [[intel::kernel_args_restrict]]{
      int total_itr = ((nx*ny)*(nz*batch))/(VFACTOR);
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
void stencil_write(queue &q, float* out, ac_int<12,true> nx, ac_int<12,true> ny, ac_int<12,true> nz,
                  ac_int<12,true> batch, double &kernel_time){
    event e3 = q.submit([&](handler &h) {
    // accessor out(out_buf, h, write_only);
    h.single_task<class stencil_write>([=] () [[intel::kernel_args_restrict]]{
      int total_itr = ((nx*ny)*(nz*batch))/(VFACTOR*2);
      const int VFACTOR2 = VFACTOR*2;
      [[intel::initiation_interval(1)]]
      for(int i = 0; i < total_itr; i++){
        struct dPath16 vec = wr_pipe::read();
        #pragma unroll VFACTOR2
        for(int v = 0; v < VFACTOR2; v++){
          out[i*VFACTOR2+v] = vec.data[v];
        }
      }
      
    });
    });

    double start0 = e3.get_profiling_info<info::event_profiling::command_start>();
    double end0 = e3.get_profiling_info<info::event_profiling::command_end>(); 
    kernel_time += (end0-start0)*1e-9;
}

template <int N, int n> struct loop {
  static void instantiate(queue &q, int nx, int ny, int nz){
    loop<N-1, n-1>::instantiate(q, nx, ny, nz);
    stencil_compute<N-1, n-1, 4096, 8>(q, nx, ny, nz);
  }
};

template<> 
struct loop<1, 1>{
  static void instantiate(queue &q, int nx, int ny, int nz){
    stencil_compute<0, 0, 4096, 8>(q, nx, ny, nz);
  }
};

// loop<90> l;


//************************************
// Vector add in DPC++ on device: returns sum in 4th parameter "sum_parallel".
//************************************
void stencil_comp(queue &q, float* input, float* output, int n_iter, int nx, int ny, int nz) {
  // Create the range object for the vectors managed by the buffer.
  int vec_size = nx*ny*nz;

  // Create buffers that hold the data shared between the host and the devices.
  // The buffer destructor is responsible to copy the data back to host when it
  // goes out of scope.

  // Submit a command group to the queue by a lambda function that contains the
  // data access permission and device computation (kernel).
  double kernel_time = 0;
  std::cout << "starting writing to the pipe\n" << std::endl;
  dpc_common::TimeInterval exe_time;

    for(int itr = 0; itr < n_iter; itr++){

      stencil_read<8>(q, input, nx, ny, nz);
      PipeConvert_512_256<8>(q,nx, ny, nz);
      loop<UFACTOR, UFACTOR>::instantiate(q, nx, ny, nz);
      PipeConvert_256_512<UFACTOR, 8>(q,nx, ny, nz);
      //write back to memory
      stencil_write<8>(q, output, nx, ny, nz, kernel_time);
      q.wait();

      
      // reading from memory
      stencil_read<8>(q, output, nx, ny, nz);
      PipeConvert_512_256<8>(q,nx, ny, nz);
      loop<UFACTOR, UFACTOR>::instantiate(q, nx, ny, nz);
      PipeConvert_256_512<UFACTOR, 8>(q,nx, ny, nz);
      //write back to memory
      stencil_write<8>(q, input, nx, ny, nz, kernel_time);
      q.wait();

    }

    std::cout << "fimished reading from the pipe\n" << std::endl;

    double exe_elapsed = exe_time.Elapsed();
    double bandwidth = 2.0*vec_size*sizeof(int)*n_iter*2.0/(kernel_time*1000000000);
    std::cout << "Elapsed time: " << kernel_time << std::endl;
    std::cout << "Bandwidth(GB/s): " << bandwidth << std::endl;
}

//************************************
// Initialize the vector from 0 to vector_size - 1
//************************************
void InitializeVector(float *a, int size) {
  for (size_t i = 0; i < size; i++) a[i] = i;
}

//************************************
// Demonstrate vector add both in sequential on CPU and in parallel on device.
//************************************
int main(int argc, char* argv[]) {

  int n_iter = 1;
  int nx = 128, ny = 128, nz=2;
  // Change vector_size if it was passed as argument
  if (argc > 1) n_iter = std::stoi(argv[1]);
  if (argc > 2) nx = std::stoi(argv[2]);
  if (argc > 3) ny = std::stoi(argv[3]);
  if (argc > 4) nz = std::stoi(argv[4]);

  nx = (nx % 8 == 0 ? nx : (nx/8+1)*8);
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

  // Create vector objects with "vector_size" to store the input and output data.
  IntVector in_vec, in_vec_h, out_sequential, out_parallel;
  in_vec.resize(nx*ny*nz);
  in_vec_h.resize(nx*ny*nz);
  out_sequential.resize(nx*ny*nz);
  out_parallel.resize(nx*ny*nz);

  // Initialize input vectors with values from 0 to vector_size - 1
  InitializeVector(in_vec);
  InitializeVector(in_vec_h);

  try {
    queue q(d_selector,  dpc_common::exception_handler, property::queue::enable_profiling{});

    float* in_vec= malloc_shared<float>(nx*ny*nz, q);
    float* out_parallel = malloc_shared<float>(nx*ny*nz, q);
    float* in_vec_h = malloc_shared<float>(nx*ny*nz, q);
    float* out_sequential = malloc_shared<float>(nx*ny*nz, q);

    InitializeVector(in_vec, nx*ny*nz);
    InitializeVector(in_vec_h,nx*ny*nz);

    // queue q2(d_selector,  dpc_common::exception_handler);


    // Print out the device information used for the kernel code.
    std::cout << "Running on device: "
              << q.get_device().get_info<info::device::name>() << "\n";
    std::cout << "Vector size: " << in_vec.size() << "\n";

    // Vector addition in DPC++
    
    stencil_comp(q, in_vec, out_parallel, n_iter, nx, ny, nz);




    for(int itr= 0; itr < UFACTOR*n_iter; itr++){
      for(int k = 0; k < nz; k++){
        for(int j = 0; j < ny; j++){
          for(int i = 0; i < nx; i++){
            int ind = k*nx*ny + j*nx + i;
             
            if(i > 0 && i < nx -1 && j > 0 && j < ny -1){
              out_sequential[ind] = in_vec_h[ind-1]*(0.125f) + in_vec_h[ind]*0.5f + \
                                       in_vec_h[ind+1]*(0.125f) + in_vec_h[ind-nx]*(0.125f) + in_vec_h[ind+nx]*(0.125f);
            } else {
              out_sequential[ind] = in_vec_h[ind];
            }
          }
        }
      }


      for(int k = 0; k < nz; k++){
        for(int j = 0; j < ny; j++){
          for(int i = 0; i < nx; i++){
            int ind = k*nx*ny + j*nx + i;
             
            if(i > 0 && i < nx -1 && j > 0 && j < ny -1){
              in_vec_h[ind] = out_sequential[ind-1]*(0.125f) + out_sequential[ind]*0.5f + \
                                 out_sequential[ind+1]*(0.125f) + out_sequential[ind-nx]*(0.125f) + out_sequential[ind+nx]*(0.125f);
            } else {
              in_vec_h[ind] = out_sequential[ind];
            }
          }
        }
      }
    }


    for(int k = 0; k < nz; k++){ 
      for(int j = 0; j < ny; j++){
        for(int i = 0; i < nx; i++){
          int ind = k*nx*ny + j*nx + i;
          float chk = fabs((in_vec_h[ind] - in_vec[ind])/(in_vec_h[ind]));
          if(chk > 0.00001 && fabs(in_vec_h[ind]) > 0.00001){
            std::cout << "j,i: " << j  << " " << i << " " << in_vec_h[ind] << " " << in_vec[ind] <<  std::endl;
            return -1;
          }
        }
      }
    }


    free(in_vec, q);
    free(out_parallel, q);
    free(in_vec_h, q);
    free(out_sequential, q);


  } catch (exception const &e) {
    std::cout << "An exception is caught for vector add.\n";
    std::terminate();
  }



  // Compute the sum of two vectors in sequential for validation.

   std::cout << "No error until here\n";

  // for (size_t i = 0; i < out_sequential.size(); i++)
  //   out_sequential.at(i) = in_vec.at(i) + 50;

  // Verify that the two vectors are equal. 


  // for (size_t i = 0; i < out_sequential.size(); i++) {
  //   if (in_vec_h.at(i) != in_vec.at(i)) {
  //     std::cout << "Vector add failed on device.\n";
  //     return -1;
  //   }
  // }


  std::cout << "Vector add successfully completed on device.\n";
  return 0;
}
