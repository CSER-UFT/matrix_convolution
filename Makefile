CC = gcc
CFLAGS = -O3 -march=native -std=c11 -D_POSIX_C_SOURCE=200809L -Wall -Wextra -Wpedantic
LDLIBS = -lm
BIN = bin
SRC = program/matrix_bench.c

PROGRAMS = \
  conv_linear_double_full conv_linear_float_full conv_linear_double_skip_kernel conv_linear_float_skip_kernel \
  conv_malloc_double_full conv_malloc_float_full conv_malloc_double_skip_kernel conv_malloc_float_skip_kernel \
  gemm_double_full gemm_float_full gemm_double_skip_k gemm_float_skip_k

VALIDATORS = $(addprefix validate_,$(PROGRAMS))

.PHONY: all validate plots clean
all: $(addprefix $(BIN)/,$(PROGRAMS))

$(BIN):
	mkdir -p $(BIN)

define build_program
$(BIN)/$(1): $(SRC) program/input_format.h | $(BIN)
	$$(CC) $$(CFLAGS) -DOPERATION=$(2) -DTECHNIQUE=$(3) $(4) -DPROGRAM_NAME='"$(1)"' -o $$@ $$(SRC) $$(LDLIBS)
endef

$(eval $(call build_program,conv_linear_double_full,CONV_LINEAR,FULL,))
$(eval $(call build_program,conv_linear_float_full,CONV_LINEAR,FULL,-DUSE_FLOAT))
$(eval $(call build_program,conv_linear_double_skip_kernel,CONV_LINEAR,APPROXIMATE,))
$(eval $(call build_program,conv_linear_float_skip_kernel,CONV_LINEAR,APPROXIMATE,-DUSE_FLOAT))
$(eval $(call build_program,conv_malloc_double_full,CONV_MALLOC,FULL,))
$(eval $(call build_program,conv_malloc_float_full,CONV_MALLOC,FULL,-DUSE_FLOAT))
$(eval $(call build_program,conv_malloc_double_skip_kernel,CONV_MALLOC,APPROXIMATE,))
$(eval $(call build_program,conv_malloc_float_skip_kernel,CONV_MALLOC,APPROXIMATE,-DUSE_FLOAT))
$(eval $(call build_program,gemm_double_full,GEMM_BLOCKED,FULL,))
$(eval $(call build_program,gemm_float_full,GEMM_BLOCKED,FULL,-DUSE_FLOAT))
$(eval $(call build_program,gemm_double_skip_k,GEMM_BLOCKED,APPROXIMATE,))
$(eval $(call build_program,gemm_float_skip_k,GEMM_BLOCKED,APPROXIMATE,-DUSE_FLOAT))

define build_validator
$(BIN)/validate_$(1): program/validate_$(2).c program/matrix_bench.c program/input_format.h | $(BIN)
	$$(CC) $$(CFLAGS) -DOPERATION=$(3) -DTECHNIQUE=$(4) $(5) -DPROGRAM_NAME='"$(1)"' -o $$@ program/validate_$(2).c $$(LDLIBS)
endef

$(eval $(call build_validator,conv_linear_double_full,linear,CONV_LINEAR,FULL,))
$(eval $(call build_validator,conv_linear_float_full,linear,CONV_LINEAR,FULL,-DUSE_FLOAT))
$(eval $(call build_validator,conv_linear_double_skip_kernel,linear,CONV_LINEAR,APPROXIMATE,))
$(eval $(call build_validator,conv_linear_float_skip_kernel,linear,CONV_LINEAR,APPROXIMATE,-DUSE_FLOAT))
$(eval $(call build_validator,conv_malloc_double_full,malloc,CONV_MALLOC,FULL,))
$(eval $(call build_validator,conv_malloc_float_full,malloc,CONV_MALLOC,FULL,-DUSE_FLOAT))
$(eval $(call build_validator,conv_malloc_double_skip_kernel,malloc,CONV_MALLOC,APPROXIMATE,))
$(eval $(call build_validator,conv_malloc_float_skip_kernel,malloc,CONV_MALLOC,APPROXIMATE,-DUSE_FLOAT))
$(eval $(call build_validator,gemm_double_full,gemm,GEMM_BLOCKED,FULL,))
$(eval $(call build_validator,gemm_float_full,gemm,GEMM_BLOCKED,FULL,-DUSE_FLOAT))
$(eval $(call build_validator,gemm_double_skip_k,gemm,GEMM_BLOCKED,APPROXIMATE,))
$(eval $(call build_validator,gemm_float_skip_k,gemm,GEMM_BLOCKED,APPROXIMATE,-DUSE_FLOAT))

validate: $(addprefix $(BIN)/,$(VALIDATORS))
	@set -e; for test in $(addprefix $(BIN)/,$(VALIDATORS)); do ./$$test; done

plots:
	python3 scripts/generate_plots.py \
		--data results/full_20260922_095004 \
		--output results/figures_full_20260922 --clean

clean:
	rm -rf $(BIN)
