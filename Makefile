CC=g++

# Configuration
DEFAULT_USERNAME ?= brandonros
# Tuned for ~100-500ms kernel dispatch (better interactivity + debugging)
GLOBAL_SIZE ?= 131072
LOCAL_SIZE ?= 64
HASHES_PER_THREAD ?= 16

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

# -ffp-contract=off prevents FMA contraction that could break determinism
# -fno-fast-math ensures strict IEEE FP semantics
# -frounding-math ensures compiler respects fesetround() rounding mode changes
CFLAGS=-c -std=c++17 -Wall -O2 -pthread -ffp-contract=off -fno-fast-math -frounding-math

# Executable
EXE=$(OUTDIR)/neurallenge

all: $(OUTDIR) $(EXE)

$(OUTDIR):
	mkdir -p $(OUTDIR)

# Generate combined kernel source (config + kernel), then convert to header
$(OUTDIR)/neurallenge_kernel.h: src/neurallenge_config.h src/neurallenge.cl | $(OUTDIR)
	cat src/neurallenge_config.h src/neurallenge.cl > $(OUTDIR)/neurallenge_combined.cl
	xxd -i $(OUTDIR)/neurallenge_combined.cl > $@
	sed -i.bak 's/output_neurallenge_combined_cl/src_neurallenge_cl/g' $@ && rm -f $@.bak

$(EXE): $(OUTDIR)/main.o
	$(CC) $< $(LDFLAGS) -o $@

$(OUTDIR)/main.o: src/main.cpp src/neurallenge_config.h $(OUTDIR)/neurallenge_kernel.h | $(OUTDIR)
	$(CC) $(CFLAGS) $(CDEFINES) -I$(OUTDIR) -Isrc $< -o $@

clean:
	rm -rf $(OUTDIR)

.PHONY: all clean
