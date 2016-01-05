
ifndef COMPILER
	MESSAGE=select a compiler to compiler in OpenMP, e.g. make COMPILER=INTEL
endif

ifeq ($(shell uname), Darwin)
	OCL = -framework OpenCL
else
	OCL = -lOpenCL
endif

CFLAGS_			= -O3
CFLAGS_INTEL	= -O3 -std=c99
CFLAGS_CRAY		= -O3
CFLAGS_GNU		= -O3 -std=c99 -lm

MPI_COMPILER 	= mpicc


snap: \
	snap_main.c \
	input.c \
	allocate.c \
	comms.c \
	problem.c \
	source.c \
	sweep.c \
	ocl_init.c \
	ocl_buffers.c \
	scalar_flux.c \
	convergence.c \
	population.c \
	profiler.c
	$(MPI_COMPILER) $^ $(CFLAGS_$(COMPILER)) $(OCL) $(OPTIONS) -o $@

kernels/%.o:	kernels/%.cu
	nvcc -c $< -o $@


KERNEL_HEADER_FILES=\
	kernels/calc_velocity_delta.o \
	kernels/calc_dd_coeff.o \
	kernels/calc_denominator.o \
	kernels/outer_source.o \
	kernels/inner_source.o \
	kernels/sweep_plane.o \
	kernels/reduce_flux.o \
	kernels/reduce_flux_moments.o

ocl_init.c:	$(KERNEL_HEADER_FILES)
	@touch ocl_init.c

.PHONY: clean

clean:
	rm -f snap kernels/*.h
