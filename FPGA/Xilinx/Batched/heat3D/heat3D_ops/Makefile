#
# The following environment variables should be predefined:
#
# OPS_INSTALL_PATH
# OPS_COMPILER (gnu,intel,etc)
#

include $(OPS_INSTALL_PATH)/../makefiles/Makefile.common
include $(OPS_INSTALL_PATH)/../makefiles/Makefile.mpi
include $(OPS_INSTALL_PATH)/../makefiles/Makefile.cuda
include $(OPS_INSTALL_PATH)/../makefiles/Makefile.hip
include $(OPS_INSTALL_PATH)/../makefiles/Makefile.hdf5

HEADERS=../heat3D_common.h ../heat3D_cpu.h

OPS_FILES=heat3D_ops_app.cpp
OPS_GENERATED=heat3D_ops_app_ops.cpp

OTHER_FILES=../heat3D_cpu.cpp 


APP=heat3D
MAIN_SRC=heat3D_ops_app

include $(OPS_INSTALL_PATH)/../makefiles/Makefile.c_app