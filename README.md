# StencilOnFPGA
Modularised HLS(Vivado C/C++) based implementation of contrasting stencil applications targetting Xilinx FPGAs. 
Implementation currently includes following applications.
- poisson (5-point 2D stencil) 
- jac2D9pt (9-point 2D stencil) 
- jac3D7pt (7-point 3D stencil) 
- RTM (multiple stencil loops with 25 point 3D stencils)
- blacksholes (3-point 1D stencil. Xilinx batch only)

Batched version targets small and medium sized meshes while Spatially blocked/Tiled version targets larger meshes which fits into 4GB buffer limit.

All the above applications are implmented using [OPS](https://github.com/OP-DSL/OPS) framework as well to support traditional architectures. 
This can be used to generate optimised target code for Muticore CPUs and GPUs.

# Acknowledgement
We would like to thank Jacques Du Toit and Tim Schmielau at NAG UK Ltd., for providing us the RTM application and for their valuable advice.
