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


const int unroll_factor = 2;
const int v_factor = 16;

struct dPath {
  [[intel::fpga_register]] float data[16];
};




// Vector type and data size for this example.
size_t vector_size = 10000;
typedef std::vector<struct dPath> IntVector; 
typedef std::vector<float> IntVectorS; 

// using rd_pipe = INTEL::pipe<class pVec16_r, dPath, 1024>;
// using wr_pipe = INTEL::pipe<class pVec16_w, dPath, 1024>;

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


struct pipeM{
  pipeM() = delete;
  template <size_t idx>  struct struct_id_M;

  template <size_t idx>
  struct PipeM{
    using pipeA = INTEL::pipe<struct_id_M<idx>, dPath, 1024>;
  };

  template <size_t idx>
  using PipeAt = typename PipeM<idx>::pipeA;
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



int copy_ToVec(float* grid_s, IntVector &grid_d1, IntVector &grid_d2, int grid_size, int delay){
  for(int i = 0; i < grid_size/(16*sizeof(float)); i++){
      for(int v = 0; v < 16; v++){
        if((i % 2) == 0){
          grid_d1[i/2+delay].data[v] = grid_s[i*16+v];
        } else {
          grid_d2[i/2+delay].data[v] = grid_s[i*16+v];
        }
      }

  }
    return 0;
}

int copy_FromVec(IntVector &grid_d1, IntVector &grid_d2,  float* grid_s, int grid_size, int delay){
  printf("grid_size:%d\n", grid_size);
  for(int i = 0; i < grid_size/(16*sizeof(float)); i++){
      for(int v = 0; v < 16; v++){
        if((i % 2) == 0){
          grid_s[i*16+v] = grid_d1[i/2+delay].data[v];
        } else  {
          grid_s[i*16+v] = grid_d2[i/2+delay].data[v];
        } 
      }

  }
    return 0;
}



// this is kind of same function for all 2D stencil applications 
// slight variant will be required when there are data strctures need to be read and write
template<int idx1, int idx2>
event stencil_read_write(queue &q, buffer<struct dPath, 1> &in_buf1, buffer<struct dPath, 1> &out_buf1,
                  buffer<struct dPath, 1> &in_buf2, buffer<struct dPath, 1> &out_buf2,
                 int total_itr , ac_int<12,true> n_iter, int delay){

      event e1 = q.submit([&](handler &h) {
      accessor in1(in_buf1, h);
      accessor out1(out_buf1, h);

      accessor in2(in_buf2, h);
      accessor out2(out_buf2, h);
      // int total_itr = ((nx*ny)*(nz))/VFACTOR;

      h.single_task<class stencil_read_write>([=] () [[intel::kernel_args_restrict]]{

      [[intel::disable_loop_pipelining]]
      for(ac_int<12,true> itr = 0; itr < n_iter; itr++){

        accessor ptrR1 = ((itr & 1) == 0) ? in1 : out1;
        accessor ptrW1 = ((itr & 1) == 1) ? in1 : out1;

        accessor ptrR2 = ((itr & 1) == 0) ? in2 : out2;
        accessor ptrW2 = ((itr & 1) == 1) ? in2 : out2;

        [[intel::ivdep]]
        [[intel::initiation_interval(1)]]
        for(int i = 0; i < total_itr+delay; i++){
          struct dPath vecR1 = ptrR1[i+delay];
          struct dPath vecR2 = ptrR2[i+delay];
          if(i < total_itr){
            pipeM::PipeAt<idx1>::write(vecR1);
            pipeM::PipeAt<idx1+1>::write(vecR2);
          }
          struct dPath vecW1;
          struct dPath vecW2;
          if(i >= delay){
            vecW1 = pipeM::PipeAt<idx2>::read();
            vecW2 = pipeM::PipeAt<idx2+1>::read();
          }
          ptrW1[i] = vecW1;
          ptrW2[i] = vecW2;
        }
      }
        
      });
      });

      return e1;
}




// this module will be same for all the designs 
template<int idx1, int idx2>
void PipeConvert_512_256(queue &q, int total_itr, ac_int<12,true> n_iter){
      ac_int<40,true> count = total_itr*n_iter;
      event e1 = q.submit([&](handler &h) {
      h.single_task<class PipeConvert_512_256>([=] () [[intel::kernel_args_restrict]]{
        [[intel::initiation_interval(1)]]
        for(ac_int<40,true> i = 0; i < count; i++){
          struct dPath data;
          if((i&1) == 0){
            data = pipeM::PipeAt<idx1>::read();
          } else {
            data = pipeM::PipeAt<idx1+1>::read();
          }
          pipeS::PipeAt<idx2>::write(data);
        }
        
      });
    });
}

template <size_t idx>  struct struct_idX;
template<int idx, int IdX, int DMAX, int VFACTOR>
void stencil_compute(queue &q, ac_int<14,true>  nx, ac_int<14,true>  ny, ac_int<14,true>  nz, int total_itr, ac_int<12,true> n_iter){
    event e2 = q.submit([&](handler &h) {
    std::string instance_name="compute"+std::to_string(idx);
    h.single_task<class struct_idX<IdX>>([=] () [[intel::kernel_args_restrict]]{
    
    
    // Parent loop: time marching loop 
    [[intel::disable_loop_pipelining]]
    for(ac_int<12,true> u_itr = 0; u_itr < n_iter; u_itr++){
      // stencil points. 2D indexes are separated by "_"
      // nonegetive index, top left index is 0_0
      struct dPath s_1_2, s_2_1, s_1_1, s_0_1, s_1_0;

      // number of words required to store a row in 3D mesh
      const int max_dpethl = DMAX/VFACTOR;

      // memory for window buffers 
      struct dPath wind1[max_dpethl]; // buffers points between s_1_2 and s_2_1
      struct dPath wind2[max_dpethl]; // buffers points between s_0_1 and s_1_0

      //vectorisation requires VFACTOR+stencil order number of elements along Xdim
      [[intel::fpga_register]] float mid_row[VFACTOR+2]; 

      // this is a variable used to implement widnow buffer 
      // it points read and write location on onchip-memory
      ac_int<14,true>  i_ld = 0;

      // variables to point x,y locations and batch id 
      short id = 0, jd = 0, kd = 0;

      // value to flag the end of the row 
      unsigned short rEnd = (nx/VFACTOR)-1;

      // forcing loop iteration to move each clock cycle 
      // by initiation interval attribute 
      [[intel::initiation_interval(1)]]
      for(int itr = 0; itr < total_itr; itr++){
        // assigning (x,y) coordinates and batch id
        // using two variables, one for codition and other for changing the state
        ac_int<14,true>  i = id; 
        ac_int<14,true>  j = jd; 
        ac_int<14,true>  k = kd;
        ac_int<14,true>  i_l = i_ld;

        // end condition for Xdim
        if(i == rEnd){
          id = 0;
        } else {
          id++;
        }


        // end condition for ydim
        if(i == rEnd && j == ny){
          jd = 1;
        } else if(i == rEnd){
          jd++;
        }

        // increment condition for batch id 
        if(i == rEnd && j == ny){
          kd++;
        }

        // reading from buffer 2
        s_1_0 = wind2[i_l];

        // updating left mid stencil point
        s_0_1 = s_1_1;

        // inputting window buffer 2
        wind2[i_l] = s_0_1;

        // updating stencil midpoint 
        s_1_1 = s_2_1;

        // reading from window buffer 1
        s_2_1 = wind1[i_l];

        // checking all the meshes have been received or not
        // inorder to avoid the blokcing pipe read call
        if(itr < (nx/VFACTOR)*ny*nz){
          s_1_2 = pipeS::PipeAt<idx>::read();
        }

        // inputing to window buffer 1 
        wind1[i_l] = s_1_2;


        // pointer update for the window buffer
        if(i_l >= nx/VFACTOR -2){
          i_ld = 0;
        } else {
          i_ld++;
        }

        // updating mid x line to vectorise the stencil computation easily 
        #pragma unroll VFACTOR
        for(int v = 0; v < VFACTOR; v++){
          mid_row[v+1] = s_1_1.data[v]; 
        }
        mid_row[0] = s_0_1.data[VFACTOR-1];
        mid_row[VFACTOR+1] = s_2_1.data[0];

        // vectored stencil computation 
        struct dPath vec_wr;
        #pragma unroll VFACTOR
        for(int v = 0; v < VFACTOR; v++){
          int i_ind = i *VFACTOR + v;
          float val =  (mid_row[v] + mid_row[v+2] + s_1_0.data[v] + s_1_2.data[v])*0.125f+ (mid_row[v+1])*0.5f;
          vec_wr.data[v] = (i_ind > 0 && i_ind < nx-1 && j > 1 && j < ny ) ? val : mid_row[v+1];
        }

        // avoiding invalid computation of intial few iterations 
        if(itr >= (nx/VFACTOR)){
          pipeS::PipeAt<idx+1>::write(vec_wr);
        }
      }
    }
  });
  });
}


// this module also same for all the designs 
template <int idx1, int idx2>
void PipeConvert_256_512(queue &q, int total_itr,  ac_int<12,true> n_iter){

    ac_int<40,true>  count = total_itr*n_iter;
    event e3 = q.submit([&](handler &h) {
    h.single_task<class pipeConvert_256_512>([=] () [[intel::kernel_args_restrict]]{
      struct dPath data16;
      [[intel::initiation_interval(1)]]
      for(ac_int<40,true>  i = 0; i < count; i++){
        struct dPath data;
        data = pipeS::PipeAt<idx1>::read();
        if((i & 1) == 0){
          pipeM::PipeAt<idx2>::write(data);
        } else{
          pipeM::PipeAt<idx2+1>::write(data);
        }
      }
    });
    });
}



template <int N, int n> struct loop {
  static void instantiate(queue &q, int nx, int ny, int nz, int total_itr, int n_iter){
    loop<N-1, n-1>::instantiate(q, nx, ny, nz, total_itr, n_iter);
    stencil_compute<N-1, n-1, 4096, 16>(q, nx, ny, nz, total_itr, n_iter);
  }
};

template<> 
struct loop<1, 1>{
  static void instantiate(queue &q, int nx, int ny, int nz, int total_itr, int n_iter){
    stencil_compute<0, 0, 4096, 16>(q, nx, ny, nz, total_itr, n_iter);
  }
};

// loop<90> l;


//************************************
// Vector add in DPC++ on device: returns sum in 4th parameter "sum_parallel".
//************************************
void stencil_comp(queue &q, IntVector &input1, IntVector &output1, 
  IntVector &input2, IntVector &output2, 
  int n_iter, int nx, int ny, int nz, int delay) {
  // Create the range object for the vectors managed by the buffer.
  range<1> num_items{input1.size()};
  int vec_size = input1.size();

  // Create buffers that hold the data shared between the host and the devices.
  // The buffer destructor is responsible to copy the data back to host when it
  // goes out of scope.
  buffer in_buf1(input1, {property::buffer::mem_channel{1}});
  buffer out_buf1(output1, {property::buffer::mem_channel{1}});
  buffer in_buf2(input2, {property::buffer::mem_channel{2}});
  buffer out_buf2(output2, {property::buffer::mem_channel{2}});

  // Submit a command group to the queue by a lambda function that contains the
  // data access permission and device computation (kernel).
  double kernel_time = 0;
  std::cout << "starting writing to the pipe\n" << std::endl;
  dpc_common::TimeInterval exe_time;


  int total_itr_32 = ((nx*ny)*(nz))/(32);
  int total_itr_16 = ((nx*ny)*(nz))/(16);
  int total_itrS = ((nx/16)*(ny*nz+1));

  // for(int itr = 0; itr < n_iter; itr++){

  // reading from memory
  event e = stencil_read_write<0, 2>(q, in_buf1, out_buf1, in_buf2, out_buf2, total_itr_32, n_iter, delay);
  PipeConvert_512_256<0,0>(q, total_itr_16, n_iter);
  loop<UFACTOR, UFACTOR>::instantiate(q, nx, ny, nz, total_itrS, n_iter);
  PipeConvert_256_512<UFACTOR,2>(q, total_itr_16, n_iter);
  q.wait();

  double start0 = e.get_profiling_info<info::event_profiling::command_start>();
  double end0 = e.get_profiling_info<info::event_profiling::command_end>(); 
  kernel_time += (end0-start0)*1e-9;

  std::cout << "fimished reading from the pipe\n" << std::endl;

  double exe_elapsed = exe_time.Elapsed();
  double bandwidth = 2.0*v_factor*vec_size*sizeof(int)*n_iter/(kernel_time*1000000000);
  std::cout << "Elapsed time: " << kernel_time << std::endl;
  std::cout << "Bandwidth(GB/s): " << bandwidth << std::endl;
}

//************************************
// Initialize the vector from 0 to vector_size - 1
//************************************
template<int VFACTOR>
void InitializeVector(IntVector &a, int delay) {
  for (size_t i = 0; i < a.size(); i++){
    for(int v = 0; v < VFACTOR; v++){
        a[i].data[v] = (i-delay)* VFACTOR + v +0.5f;
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
  int nx = 128, ny = 128, nz=2;
  // Change vector_size if it was passed as argument
  if (argc > 1) n_iter = std::stoi(argv[1]);
  if (argc > 2) nx = std::stoi(argv[2]);
  if (argc > 3) ny = std::stoi(argv[3]);
  if (argc > 4) nz = std::stoi(argv[4]);

  nx = (nx % 32 == 0 ? nx : (nx/32+1)*32);
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

  int delay = (nx/(v_factor*2))*UFACTOR+839 + (4+10)/2 + (33*UFACTOR)/2;
  IntVectorS in_vec_h, out_sequential, out_vec_d;
  in_vec_h.resize(nx*ny*nz);
  out_sequential.resize(nx*ny*nz);
  out_vec_d.resize(nx*ny*nz);
  

  // Initialize input vectors with values from 0 to vector_size - 1

  IntVector in_vec_1, out_parallel_1;
  IntVector in_vec_2, out_parallel_2;

  in_vec_1.resize(nx/v_factor*ny*nz+delay*2);
  out_parallel_1.resize(nx/v_factor*ny*nz+delay*2);

  in_vec_2.resize(nx/v_factor*ny*nz+delay*2);
  out_parallel_2.resize(nx/v_factor*ny*nz+delay*2);

  // InitializeVector<v_factor>(in_vec, delay);
  InitializeVectorS(in_vec_h);
  copy_ToVec(in_vec_h.data(), in_vec_1, in_vec_2, nx*ny*nz*sizeof(float), delay);
  try {
    queue q(d_selector,  dpc_common::exception_handler, property::queue::enable_profiling{});

    // queue q2(d_selector,  dpc_common::exception_handler);


    // Print out the device information used for the kernel code.
    std::cout << "Running on device: "
              << q.get_device().get_info<info::device::name>() << "\n";
    std::cout << "Vector size: " << in_vec_1.size() << "\n";

    // Vector addition in DPC++
    
    stencil_comp(q, in_vec_1, out_parallel_1,  in_vec_2, out_parallel_2, 2*n_iter, nx, ny, nz, delay);

  } catch (exception const &e) {
    std::cout << "An exception is caught for vector add.\n";
    std::terminate();
  }

  copy_FromVec(out_parallel_1, out_parallel_2, out_vec_d.data(), nx*ny*nz*sizeof(float), delay);


  // Compute the sum of two vectors in sequential for validation.

  for(int itr= 0; itr < UFACTOR*n_iter; itr++){
    for(int k = 0; k < nz; k++){
      for(int j = 0; j < ny; j++){
        for(int i = 0; i < nx; i++){
          int ind = k*nx*ny + j*nx + i;
           
          if(i > 0 && i < nx -1 && j > 0 && j < ny -1){
            out_sequential.at(ind) = in_vec_h.at(ind-1)*(0.125f) + in_vec_h.at(ind)*0.5f + in_vec_h.at(ind+1)*(0.125f) + in_vec_h.at(ind-nx)*(0.125f) + in_vec_h.at(ind+nx)*(0.125f);
          } else {
            out_sequential.at(ind) = in_vec_h.at(ind);
          }
        }
      }
    }


    for(int k = 0; k < nz; k++){
      for(int j = 0; j < ny; j++){
        for(int i = 0; i < nx; i++){
          int ind = k*nx*ny + j*nx + i;
           
          if(i > 0 && i < nx -1 && j > 0 && j < ny -1){
            in_vec_h.at(ind) = out_sequential.at(ind-1)*(0.125f) + out_sequential.at(ind)*0.5f + out_sequential.at(ind+1)*(0.125f) + out_sequential.at(ind-nx)*(0.125f) + out_sequential.at(ind+nx)*(0.125f);
          } else {
            in_vec_h.at(ind) = out_sequential.at(ind);
          }
        }
      }
    }



  }
   std::cout << "No error until here\n";

  // for (size_t i = 0; i < out_sequential.size(); i++)
  //   out_sequential.at(i) = in_vec.at(i) + 50;

  // Verify that the two vectors are equal. 
  for(int k = 0; k < nz; k++){ 
    for(int j = 0; j < ny; j++){
      for(int i = 0; i < nx; i++){
          int ind = k*nx*ny + j*nx + i;
          float chk = fabs((in_vec_h.at(ind) - out_vec_d.at(ind))/(in_vec_h.at(ind)));
          if(chk > 0.00001 && fabs(in_vec_h.at(ind)) > 0.00001){
            std::cout << "j,i, k, ind: " << j  << " " << i << " " << k << " " << ind << " " << in_vec_h.at(ind) << " " << out_vec_d.at(ind) <<  std::endl;
            return -1;
          }
      }
    }
  }

  in_vec_h.clear();
  out_sequential.clear();

  in_vec_1.clear();
  out_parallel_1.clear();
  in_vec_2.clear();
  out_parallel_2.clear();

  std::cout << "Vector add successfully completed on device.\n";
  return 0;
}
