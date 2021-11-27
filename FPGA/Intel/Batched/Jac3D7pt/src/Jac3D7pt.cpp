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
#endif

using namespace sycl;

// Vector type and data size for this example.
size_t vector_size = 10000;
typedef std::vector<float> IntVector; 
const int unroll_factor = 2;

struct dPath {
  [[intel::fpga_register]] float data[8];
};

using rd_pipe = INTEL::pipe<class pVec8, dPath, 8000000>;
using wr_pipe = INTEL::pipe<class pVec8, dPath, 8000000>;

#define UFACTOR 1

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




void stencil_read(queue &q, buffer<float, 1> &in_buf, int nx, int ny, int nz, int batch){
      event e1 = q.submit([&](handler &h) {
      accessor in(in_buf, h, read_only);
      h.single_task<class producer>([=] () [[intel::kernel_args_restrict]]{

        [[intel::initiation_interval(1)]]
        for(int i = 0; i < (nx*ny*nz*batch)/8; i++){
          struct dPath vec;
          #pragma unroll 8
          for(int v = 0; v < 8; v++){
            vec.data[v] = in[i*8+v];
          }
          pipeS::PipeAt<0>::write(vec);
        }
        
      });
      });
}

template <size_t idx>  struct struct_idX;
template<int idx, int IdX>
void stencil_compute(queue &q, unsigned short nx, unsigned short ny, unsigned short nz, unsigned short batch){
    event e2 = q.submit([&](handler &h) {

    std::string instance_name="compute"+std::to_string(idx);
    h.single_task<class struct_idX<IdX>>([=] () [[intel::kernel_args_restrict]]{
    int total_itr = ((nx>>3)*ny*(batch*nz+1));

    const int max_dpethl = 256/8;
    const int max_dpethP = 256*256/8;
    const int VFACTOR = 8;
    struct dPath s_1_1_2, s_1_2_1, s_1_1_1, s_1_1_1_b, s_1_1_1_f, s_1_0_1, s_1_1_0;
    struct dPath window_1[max_dpethP], window_2[max_dpethl], window_3[max_dpethl], window_4[max_dpethP];



    struct dPath vec_wr;
    [[intel::fpga_register]] float mid_row[VFACTOR+2];
    unsigned short j_l = 0, j_p = 0;


    short id = 0, jd = 0, kd = 0, batd = 0;;
    unsigned int mesh_size = (nx*ny)>>3;
    unsigned short rEnd = (nx>>3)-1;

    [[intel::initiation_interval(1)]]
    for(int itr = 0; itr < total_itr; itr++){
      unsigned short i = id; // itr % rEnd; //id;
      unsigned short j = jd; //itr / rEnd ;///jd;
      unsigned short k = kd;
      unsigned short bat = batd;

      if(i == rEnd){
        id = 0;
      } else {
        id++;
      }

      if(i == rEnd && j == ny-1){
        jd = 0;
      } else if(i == rEnd){
        jd++;
      }


      if(i == rEnd && j == ny-1 && k == nz){
        kd = 1;
      }else if(i == rEnd && j == ny-1){
        kd++;
      }


      s_1_1_0 = window_4[j_p];

      s_1_0_1 = window_3[j_l];
      window_4[j_p] = s_1_0_1;

      s_1_1_1_b = s_1_1_1;
      window_3[j_l] = s_1_1_1_b;

      s_1_1_1 = s_1_1_1_f;
      s_1_1_1_f = window_2[j_l];  // read

      s_1_2_1 = window_1[j_p];   // read
      window_2[j_l] = s_1_2_1;  //set

      if(itr < (nx>>3)*ny*nz*batch){
        s_1_1_2 = pipeS::PipeAt<idx>::read();
      }

      window_1[j_p] = s_1_1_2;

      j_l++;
      if(j_l >= nx/8 -1){
        j_l = 0;
      }

      j_p++;
      if(j_p >= (nx/8)*(ny-1)){
        j_p = 0;
      }

      #pragma unroll 8
      for(int v = 0; v < VFACTOR; v++){
        mid_row[v+1] = s_1_1_1.data[v]; 
      }

      mid_row[0] = s_1_1_1_b.data[VFACTOR-1];
      mid_row[VFACTOR+1] = s_1_1_1_f.data[0];

      // #pragma unroll 8
      process: for(short q = 0; q < VFACTOR; q++){
        short index = (i << 3) + q;
        float r1_1_2 =  s_1_1_2.data[q] * (0.02f);
        float r1_2_1 =  s_1_2_1.data[q] * (0.04f);
        float r0_1_1 =  mid_row[q] * (0.05f);
        float r1_1_1 =  mid_row[q+1] * (0.79f);
        float r2_1_1 =  mid_row[q+2] * (0.06f);
        float r1_0_1 =  s_1_0_1.data[q] * (0.03f);
        float r1_1_0 =  s_1_1_0.data[q] * (0.01f);

        float f1 = r1_1_2 + r1_2_1;
        float f2 = r0_1_1 + r1_1_1;
        float f3 = r2_1_1 + r1_0_1;


        float r1 = f1 + f2;
        float r2=  f3 + r1_1_0;

        float result  = r1 + r2;
        bool change_cond = (index <= 0 || index >= nx-1 || (k <= 1) || (k >= nz) || (j <= 0) || (j >= ny -1));
        vec_wr.data[q] = change_cond ? mid_row[q+1] : result;
      }

      bool cond_wr = (k >= 1) && ( k < nz+1);

      // if(itr < (nx>>3)*ny*nz){
      if(itr >= (nx>>3)*ny){
        pipeS::PipeAt<idx+1>::write(vec_wr);
      }
    }
    
  });
  });
}

template <int idx>
void stencil_write(queue &q, buffer<float, 1> &out_buf, int nx, int ny, int nz, int batch, double &kernel_time){
    event e3 = q.submit([&](handler &h) {
    accessor out(out_buf, h, write_only);
    std::string instance_name="consumer"+std::to_string(idx);
    h.single_task<class instance_name>([=] () [[intel::kernel_args_restrict]]{
      [[intel::initiation_interval(1)]]

      for(int i = 0; i < (nx*ny*nz*batch)/8; i++){
        struct dPath vec = pipeS::PipeAt<idx>::read();
        #pragma unroll 8
        for(int v = 0; v < 8; v++){
          out[i*8+v] = vec.data[v];
        }
      }
      
    });
    });

    double start0 = e3.get_profiling_info<info::event_profiling::command_start>();
    double end0 = e3.get_profiling_info<info::event_profiling::command_end>(); 
    kernel_time += (end0-start0)*1e-9;
}


template <int N, int n> struct loop {
  static void instantiate(queue &q, int nx, int ny, int nz, int batch){
    loop<N-1, n-1>::instantiate(q, nx, ny, nz, batch);
    stencil_compute<N-1, n-1>(q, nx, ny, nz, batch);
  }
};

template<> 
struct loop<1, 1>{
  static void instantiate(queue &q, int nx, int ny, int nz, int batch){
    stencil_compute<0, 0>(q, nx, ny, nz, batch);
  }
};

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

    for(int itr = 0; itr < n_iter; itr++){

      // reading from memory
      stencil_read(q, in_buf, nx, ny, nz, batch);
      loop<UFACTOR, UFACTOR>::instantiate(q, nx, ny, nz, batch);
      //write back to memory
      stencil_write<UFACTOR>(q, out_buf, nx, ny, nz, batch, kernel_time);
      q.wait();

      
      // reading from memory
      stencil_read(q, out_buf, nx, ny, nz, batch);
      // computation
      loop<UFACTOR, UFACTOR>::instantiate(q, nx, ny, nz, batch);
      //write back to memory
      stencil_write<UFACTOR>(q, in_buf, nx, ny, nz, batch, kernel_time);
      

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
void InitializeVector(IntVector &a) {
  for (size_t i = 0; i < a.size(); i++) a.at(i) = i;
}

//************************************
// Demonstrate vector add both in sequential on CPU and in parallel on device.
//************************************
int main(int argc, char* argv[]) {

  int n_iter = 1;
  int nx = 8, ny = 8, nz=4, batch = 4;
  // Change vector_size if it was passed as argument
  if (argc > 1) n_iter = std::stoi(argv[1]);
  if (argc > 2) nx = std::stoi(argv[2]);
  if (argc > 3) ny = std::stoi(argv[3]);
  if (argc > 4) nz = std::stoi(argv[4]);
  if (argc > 5) batch = std::stoi(argv[5]);

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
  in_vec.resize(nx*ny*nz*batch);
  in_vec_h.resize(nx*ny*nz*batch);
  out_sequential.resize(nx*ny*nz*batch);
  out_parallel.resize(nx*ny*nz*batch);

  // Initialize input vectors with values from 0 to vector_size - 1
  InitializeVector(in_vec);
  InitializeVector(in_vec_h);

  try {
    queue q(d_selector,  dpc_common::exception_handler, property::queue::enable_profiling{});

    // queue q2(d_selector,  dpc_common::exception_handler);


    // Print out the device information used for the kernel code.
    std::cout << "Running on device: "
              << q.get_device().get_info<info::device::name>() << "\n";
    std::cout << "Vector size: " << in_vec.size() << "\n";

    // Vector addition in DPC++
    
    stencil_comp(q, in_vec, out_parallel, n_iter, nx, ny, nz, batch);

  } catch (exception const &e) {
    std::cout << "An exception is caught for vector add.\n";
    std::terminate();
  }



  // Compute the sum of two vectors in sequential for validation.

  for(int itr= 0; itr < UFACTOR*n_iter; itr++){
    for(int bat = 0; bat < batch; bat++){
      for(int k = 0; k < nz; k++){
        for(int j = 0; j < ny; j++){
          for(int i = 0; i < nx; i++){
            int ind = k*nx*ny + j*nx + i;
            int offset = bat*nx*ny*nz;
            if(i > 0 && i < nx -1 && j > 0 && j < ny -1 && k > 0 && k < nz-1){
              out_sequential.at(offset+ind) =   in_vec_h.at(offset+ind-nx*ny)*(0.01f) + in_vec_h.at(offset+ind+nx*ny)*0.02f + in_vec_h.at(offset+ind-nx)*(0.03f) + \
                                                in_vec_h.at(offset+ind+nx)*(0.04f) + in_vec_h.at(offset+ind-1)*(0.05f) + in_vec_h.at(offset+ind+1)*(0.06f) \
                                                + in_vec_h.at(offset+ind)*(0.79f);
            } else {
              out_sequential.at(offset+ind) = in_vec_h.at(offset+ind);
            }
          }
        }
      }
    }

    for(int bat = 0; bat < batch; bat++){
      for(int k = 0; k < nz; k++){
        for(int j = 0; j < ny; j++){
          for(int i = 0; i < nx; i++){
            int ind = k*nx*ny + j*nx + i;
            int offset = bat*nx*ny*nz;
            if(i > 0 && i < nx -1 && j > 0 && j < ny -1 && k > 0 && k < nz-1){
              in_vec_h.at(offset+ind) =  out_sequential.at(offset+ind-nx*ny)*(0.01f) + out_sequential.at(offset+ind+nx*ny)*0.02f + out_sequential.at(offset+ind-nx)*(0.03f) + \
                                         out_sequential.at(offset+ind+nx)*(0.04f) + out_sequential.at(offset+ind-1)*(0.05f) + out_sequential.at(offset+ind+1)*(0.06f) \
                                         + out_sequential.at(offset+ind)*(0.79f);

            } else {
              in_vec_h.at(offset+ind) = out_sequential.at(offset+ind);
            }
          }
        }
      }
    }



  }
   std::cout << "No error until here\n";

  // for (size_t i = 0; i < out_sequential.size(); i++)
  //   out_sequential.at(i) = in_vec.at(i) + 50;

  // Verify that the two vectors are equal. 
  for(int bat = 0; bat < batch; bat++){
    for(int k = 0; k < nz; k++){ 
      for(int j = 0; j < ny; j++){
        for(int i = 0; i < nx; i++){
          int ind = bat*nx*ny*nz + k*nx*ny + j*nx + i;
          float chk = fabs((in_vec_h.at(ind) - in_vec.at(ind))/(in_vec_h.at(ind)));
          if(chk > 0.00001 && fabs(in_vec_h.at(ind)) > 0.00001){
            std::cout << "j,i: " << j  << " " << i << " " << in_vec_h.at(ind) << " " << in_vec.at(ind) <<  std::endl;
            return -1;
          }
        }
      }
    }
  }



  int indices[]{0, 1, 2, (static_cast<int>(in_vec.size()) - 1)};
  constexpr size_t indices_size = sizeof(indices) / sizeof(int);

  // Print out the result of vector add.
  for (int i = 0; i < indices_size; i++) {
    int j = indices[i];
    if (i == indices_size - 1) std::cout << "...\n";
    std::cout << "[" << j << "]: " << in_vec[j] << " + 50 = "
              << out_parallel[j] << "\n";
  }

  in_vec.clear();
  out_sequential.clear();
  out_parallel.clear();

  std::cout << "Vector add successfully completed on device.\n";
  return 0;
}
