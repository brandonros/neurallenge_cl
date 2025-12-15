CC=g++

# Configuration - Tuned for RTX 5090 (4x GPU setup)
# GLOBAL_SIZE: Total work-items per kernel launch (per GPU)
#   - Higher = more parallelism, but longer kernel time
#   - RTX 5090 has 21760 CUDA cores, can handle 200K+ concurrent threads
# LOCAL_SIZE: Work-items per work-group
#   - Must divide GLOBAL_SIZE evenly
#   - 128-256 typically optimal for modern NVIDIA GPUs
# HASHES_PER_THREAD: Nonces evaluated per work-item
#   - Higher = better weight cache reuse, fewer kernel launches
#   - Too high = longer kernel time, less responsive target updates
DEFAULT_USERNAME ?= brandonros
GLOBAL_SIZE ?= 131072
LOCAL_SIZE ?= 128
HASHES_PER_THREAD ?= 128

CDEFINES=-DCL_TARGET_OPENCL_VERSION=300 \
         -DDEFAULT_USERNAME=\"$(DEFAULT_USERNAME)\" \
         -DGLOBAL_SIZE=$(GLOBAL_SIZE) \
         -DLOCAL_SIZE=$(LOCAL_SIZE) \
         -DHASHES_PER_THREAD=$(HASHES_PER_THREAD)

OUTDIR=output

# Platform-specific flags
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
    LDFLAGS=-framework OpenCL -pthread
else
    LDFLAGS=-lOpenCL -pthread
endif

# Host compiler flags:
# -O3: Aggressive optimization for host code
# -march=native: Use CPU-specific instructions (only affects host, not GPU)
# -ffp-contract=off: Prevents FMA contraction that could break determinism
# -fno-fast-math: Ensures strict IEEE FP semantics
# -frounding-math: Ensures compiler respects fesetround() rounding mode changes
CFLAGS=-c -std=c++17 -Wall -O3 -march=native -pthread -ffp-contract=off -fno-fast-math -frounding-math

# Executable
EXE=$(OUTDIR)/neurallenge

all: $(OUTDIR) $(EXE)

$(OUTDIR):
	mkdir -p $(OUTDIR)

# Generate combined kernel source (config + kernel), then convert to header
$(OUTDIR)/kernel_embedded.h: src/config.h src/kernel.cl | $(OUTDIR)
	cat src/config.h src/kernel.cl > $(OUTDIR)/kernel_combined.cl
	xxd -i $(OUTDIR)/kernel_combined.cl > $@
	sed -i.bak 's/output_kernel_combined_cl/kernel_cl/g' $@ && rm -f $@.bak

$(EXE): $(OUTDIR)/main.o
	$(CC) $< $(LDFLAGS) -o $@

$(OUTDIR)/main.o: src/main.cpp src/config.h $(OUTDIR)/kernel_embedded.h | $(OUTDIR)
	$(CC) $(CFLAGS) $(CDEFINES) -I$(OUTDIR) -Isrc $< -o $@

clean:
	rm -rf $(OUTDIR)

.PHONY: all clean
