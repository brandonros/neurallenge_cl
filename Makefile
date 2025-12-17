CC=g++

# Configuration - Tuned for Apple M4
# GLOBAL_SIZE: Total work-items per kernel launch (per GPU)
#   - M4 has ~10 GPU cores (~1280 ALUs) - much smaller than discrete GPUs
# LOCAL_SIZE: Work-items per work-group
#   - Apple GPUs prefer smaller workgroups (64 typical)
# HASHES_PER_THREAD: Nonces evaluated per work-item
#   - Higher = better weight cache reuse, fewer kernel launches
#   - Too high = longer kernel time, less responsive target updates
DEFAULT_USERNAME ?= brandonros
GLOBAL_SIZE ?= 8192
LOCAL_SIZE ?= 64
HASHES_PER_THREAD ?= 64

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
    LDFLAGS_SERVER=-framework OpenCL -lsqlite3 -pthread
else
    LDFLAGS=-lOpenCL -pthread
    LDFLAGS_SERVER=-lOpenCL -lsqlite3 -pthread
endif

# Host compiler flags:
# -O3: Aggressive optimization for host code
# -march=native: Use CPU-specific instructions (only affects host, not GPU)
# -ffp-contract=off: Prevents FMA contraction that could break determinism
# -fno-fast-math: Ensures strict IEEE FP semantics
# -frounding-math: Ensures compiler respects fesetround() rounding mode changes
CFLAGS=-c -std=c++17 -Wall -O3 -march=native -pthread -ffp-contract=off -fno-fast-math -frounding-math

# Executables
MINER=$(OUTDIR)/miner
SERVER=$(OUTDIR)/server

all: $(OUTDIR) $(MINER) $(SERVER)

miner: $(MINER)

server: $(SERVER)

$(OUTDIR):
	mkdir -p $(OUTDIR)

# Generate combined kernel source (config + kernel), then convert to header
$(OUTDIR)/kernel_embedded.h: src/config.h src/kernel.cl | $(OUTDIR)
	cat src/config.h src/kernel.cl > $(OUTDIR)/kernel_combined.cl
	xxd -i $(OUTDIR)/kernel_combined.cl > $@
	sed -i.bak 's/output_kernel_combined_cl/kernel_cl/g' $@ && rm -f $@.bak

# Miner (GPU)
$(MINER): $(OUTDIR)/miner.o
	$(CC) $< $(LDFLAGS) -o $@

$(OUTDIR)/miner.o: src/miner.cpp src/config.h $(OUTDIR)/kernel_embedded.h | $(OUTDIR)
	$(CC) $(CFLAGS) $(CDEFINES) -I$(OUTDIR) -Isrc -Ivendor $< -o $@

# Server (HTTP + OpenCL verifier + SQLite)
$(SERVER): $(OUTDIR)/server.o
	$(CC) $< $(LDFLAGS_SERVER) -o $@

$(OUTDIR)/server.o: src/server.cpp src/config.h $(OUTDIR)/kernel_embedded.h | $(OUTDIR)
	$(CC) $(CFLAGS) $(CDEFINES) -I$(OUTDIR) -Isrc -Ivendor $< -o $@

clean:
	rm -rf $(OUTDIR)

.PHONY: all clean miner server
