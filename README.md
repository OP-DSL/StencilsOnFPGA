# StencilsOnFPGA
Modularised HLS based implementation of contrasting stencil applications targetting Xilinx FPGAs. 
implementation currently includes following applications
-Poisson (5-point 2D stencil)
-jac2D9pt (9-point 2D stencil)
-jac3D7pt (7-point 3D stencil)
-RTM (multiple stencil loops with 25 point 3D stencil)

Kernels are designed using Vivado C/C++ but can be converted to OpenCL using same methodology. 
Validated on Xilinx U280 device. 
