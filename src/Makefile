
ifndef COMPILER
	MESSAGE=select a compiler to compiler in OpenMP, e.g. make COMPILER=INTEL
endif

CUDA = -lcudart

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
	problem.o \
	source.o \
	sweep.o \
	cuda_buffers.c \
	scalar_flux.o \
	convergence.c \
	population.c \
	profiler.c \
	cuda_error.c
	$(MPI_COMPILER) $^ $(CFLAGS_$(COMPILER)) $(CUDA) $(OPTIONS) -o $@

%.o:	%.cu
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

snap:	$(KERNEL_HEADER_FILES)

.PHONY: clean

clean:
	rm -f snap kernels/*.o
