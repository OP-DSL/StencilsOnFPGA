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

HEADERS=../blacksholes_common.h ../blacksholes_cpu.h 

OPS_FILES=blacksholes_ops_app.cpp
OPS_GENERATED=blacksholes_ops_app_ops.cpp

OTHER_FILES=../blacksholes_cpu.cpp 


APP=blacksholes
MAIN_SRC=blacksholes_ops_app

include $(OPS_INSTALL_PATH)/../makefiles/Makefile.c_app