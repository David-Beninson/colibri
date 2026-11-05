# Colibri Build System — Portable CPU default (optional macOS + Metal)

ifeq ($(shell uname -s 2>/dev/null),Darwin)
ifneq ($(wildcard /Library/Developer/CommandLineTools),)
export DEVELOPER_DIR ?= /Library/Developer/CommandLineTools
endif
endif

CC            ?= cc
CXX           ?= c++
PYTHON        ?= python3
METAL         ?= 0
EXTRA_CFLAGS  ?=
EXTRA_LDFLAGS ?=

PREFIX        ?= /usr/local
BINDIR        ?= $(PREFIX)/bin
LIBEXECDIR    ?= $(PREFIX)/libexec/colibri
INSTALL       ?= install

# --- OpenMP detection (Homebrew / MacPorts) ---
OMP_EXPLICIT := $(filter command line environment,$(origin OMPDIR))
ifeq ($(OMP_EXPLICIT),)
OMPDIR := $(shell brew --prefix libomp 2>/dev/null)
ifeq ($(OMPDIR),)
OMPDIR := $(firstword $(wildcard /opt/homebrew/opt/libomp /usr/local/opt/libomp))
endif
endif

ifneq ($(and $(OMPDIR),$(wildcard $(OMPDIR)/include/omp.h),$(wildcard $(OMPDIR)/lib/libomp.*)),)
OMPINC := $(OMPDIR)/include
OMPLIB := $(OMPDIR)/lib
else ifeq ($(OMP_EXPLICIT),)
ifneq ($(and $(wildcard /opt/local/include/libomp/omp.h),$(wildcard /opt/local/lib/libomp/libomp.*)),)
OMPINC := /opt/local/include/libomp
OMPLIB := /opt/local/lib/libomp
endif
endif

ifneq ($(OMPINC),)
OMPC = -Xclang -fopenmp -I$(OMPINC)
OMPL = -L$(OMPLIB) -lomp
else
OMPC =
OMPL =
endif

# --- Compiler & Linker Flags ---
WARN_FLAGS = -Wall -Wextra -Wno-unused-parameter -Wno-misleading-indentation -Wno-unused-function -Wno-deprecated-declarations -MMD -MP

CFLAGS     = -O3 $(WARN_FLAGS) $(EXTRA_CFLAGS)
LDFLAGS    = -lm $(EXTRA_LDFLAGS)

ifneq ($(ARCH),)
CFLAGS    += -mcpu=$(ARCH)
endif

ifneq ($(OMPINC),)
CFLAGS    += $(OMPC)
LDFLAGS   += $(OMPL)
endif

ifeq ($(METAL),1)
SYSROOT ?= $(shell xcrun --show-sdk-path 2>/dev/null)
ifeq ($(SYSROOT),)
SYSROOT := $(wildcard /Library/Developer/CommandLineTools/SDKs/MacOSX.sdk)
endif
SYSROOT_FLAG := $(if $(SYSROOT),-isysroot $(SYSROOT),)

METAL_FRAMEWORKS = -framework Metal -framework MetalPerformanceShaders -framework Accelerate -framework Foundation
CFLAGS          += -DCOLI_METAL -DQWEN36_QPACK -DCOLI_ACCELERATE -DACCELERATE_NEW_LAPACK $(SYSROOT_FLAG)
LDFLAGS         += $(METAL_FRAMEWORKS) $(SYSROOT_FLAG) -lc++
METAL_CXX        = $(CXX) $(SYSROOT_FLAG) -x objective-c++ -std=gnu++17 -fobjc-arc -O3 -DACCELERATE_NEW_LAPACK $(if $(ARCH),-mcpu=$(ARCH),)
METALXX          = $(METAL_CXX) $(METAL_FRAMEWORKS)
QWEN36_OBJS      = backend_metal.o qwen36_qpack.o qpack.o
else
QWEN36_OBJS      =
endif

ifeq ($(LTO),1)
CFLAGS  += -flto
LDFLAGS += -flto
endif

# --- Core Targets ---
.PHONY: all metal-test qwen36-metal-test gemm-test test check clean install uninstall

all: qwen36

backend_metal.o: backend_metal.mm backend_metal.h affine_quant.h
	$(METAL_CXX) $(WARN_FLAGS) -c backend_metal.mm -o $@

qwen36_qpack.o: qwen36_qpack.c qwen36_qpack.h qpack.h affine_quant.h backend_metal.h
	$(CC) $(CFLAGS) -c qwen36_qpack.c -o $@

qpack.o: qpack.c qpack.h compat.h
	$(CC) $(CFLAGS) -c qpack.c -o $@

qwen36: qwen36.c $(QWEN36_OBJS)
	$(CC) $(CFLAGS) qwen36.c $(QWEN36_OBJS) -o $@ $(LDFLAGS)

qwen36-canonical: tools/qwen36_canonical.c qwen36.c qwen36_dense_file.h
	$(CC) $(CFLAGS) tools/qwen36_canonical.c -o $@ $(LDFLAGS)

# --- Test Targets ---
metal-test: tests/test_backend_metal.mm backend_metal.mm backend_metal.h affine_quant.h
	$(METALXX) tests/test_backend_metal.mm backend_metal.mm -o backend_metal_test
	./backend_metal_test

gemm-test: tests/test_gemm_largebatch.mm backend_metal.mm backend_metal.h affine_quant.h
	$(METALXX) $(OMPC) tests/test_gemm_largebatch.mm backend_metal.mm $(OMPL) -o gemm_largebatch_test
	./gemm_largebatch_test

qwen36_qpack_metal_test: tests/test_qwen36_qpack_metal.c qwen36.c qwen36_qpack.o qpack.o backend_metal.o
	$(CC) $(CFLAGS) -DCOLI_METAL -DQWEN36_QPACK tests/test_qwen36_qpack_metal.c qwen36_qpack.o qpack.o backend_metal.o -o $@ $(METAL_FRAMEWORKS) -lc++ $(LDFLAGS)

qwen36-metal-test: qwen36_qpack_metal_test
	./qwen36_qpack_metal_test

tests/test_qpack: tests/test_qpack.c qpack.o
	$(CC) $(CFLAGS) tests/test_qpack.c qpack.o -o $@ $(LDFLAGS)

tests/test_kv16: tests/test_kv16.c kv16.h
	$(CC) $(CFLAGS) tests/test_kv16.c -o $@ $(LDFLAGS)

tests/test_kv_fp8: tests/test_kv_fp8.c kv_fp8.h decode_batch.h
	$(CC) $(CFLAGS) tests/test_kv_fp8.c -o $@ $(LDFLAGS)

tests/test_gguf_c: tests/test_gguf_c.c gguf.h
	$(CC) $(CFLAGS) tests/test_gguf_c.c -o $@ $(LDFLAGS)

tests/test_ggml_quants: tests/test_ggml_quants.c ggml_quants.h
	$(CC) $(CFLAGS) tests/test_ggml_quants.c -o $@ $(LDFLAGS)

test: tests/test_qpack tests/test_kv16 tests/test_kv_fp8
	./tests/test_qpack
	./tests/test_kv16
	./tests/test_kv_fp8

check: clean qwen36 test

# --- Install & Clean ---
install: qwen36
	$(INSTALL) -d $(DESTDIR)$(BINDIR)
	$(INSTALL) -d $(DESTDIR)$(LIBEXECDIR)
	$(INSTALL) -d $(DESTDIR)$(LIBEXECDIR)/tools
	$(INSTALL) -m 755 coli $(DESTDIR)$(BINDIR)/coli
	$(INSTALL) -m 755 qwen36 $(DESTDIR)$(LIBEXECDIR)/qwen36
	$(INSTALL) -m 644 *.py $(DESTDIR)$(LIBEXECDIR)/
	$(INSTALL) -m 644 tools/*.py $(DESTDIR)$(LIBEXECDIR)/tools/

uninstall:
	rm -f $(DESTDIR)$(BINDIR)/coli
	rm -rf $(DESTDIR)$(LIBEXECDIR)

clean:
	rm -rf qwen36 qwen36-canonical backend_metal_test gemm_largebatch_test qwen36_qpack_metal_test \
	       *.o *.d *.dSYM tests/*.o tests/*.d tests/*.dSYM \
	       tests/test_qpack tests/test_kv16 tests/test_kv_fp8 tests/test_gguf_c tests/test_ggml_quants

-include *.d tests/*.d
