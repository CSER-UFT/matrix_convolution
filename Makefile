# Compilação dos benchmarks do TCC2.
#
# Cada operação tem um binário por precisão; a versão completa e as aproximadas
# usam o mesmo binário (parâmetro --step em tempo de execução).
#
# MARCH=native otimiza para a CPU de onde o make roda. Por isso o
# run_benchmark.sh recompila dentro do job, no nó que executa a campanha.
# Para binários portáveis entre nós diferentes use, por exemplo:
#   make MARCH=x86-64-v3
#
# Sem -ffast-math de propósito: ele permitiria reordenar somas em ponto
# flutuante e misturaria um erro numérico do compilador com o erro das técnicas
# aproximadas, que é justamente o que o trabalho mede.

CC      = gcc
MARCH  ?= native
CFLAGS  = -O3 -march=$(MARCH) -g -std=c11 -D_POSIX_C_SOURCE=200809L -Wall -Wextra -Wpedantic
LDLIBS  = -lm
BIN     = bin
SRC     = program/matrix_bench.c
HDR     = program/input_format.h
VALSRC  = program/validate.c

# Diretórios usados pelo alvo plots (podem ser trocados na linha de comando).
DATA    ?= results/analysis_data
FIGURES ?= results/figures

PROGRAMS = conv_linear_double conv_linear_float \
           conv_malloc_double conv_malloc_float \
           gemm_double gemm_float

VALIDATORS = $(addprefix validate_,$(PROGRAMS))

.PHONY: all validate vecreport compiler-info plots clean
all: $(addprefix $(BIN)/,$(PROGRAMS))

$(BIN):
	mkdir -p $(BIN)

# $(1) nome, $(2) operação, $(3) flags extras
define build_program
$(BIN)/$(1): $(SRC) $(HDR) | $(BIN)
	$$(CC) $$(CFLAGS) -DOPERATION=$(2) $(3) -DPROGRAM_NAME='"$(1)"' -o $$@ $$(SRC) $$(LDLIBS)

$(BIN)/validate_$(1): $(VALSRC) $(SRC) $(HDR) | $(BIN)
	$$(CC) $$(CFLAGS) -DOPERATION=$(2) $(3) -DPROGRAM_NAME='"$(1)"' -o $$@ $$(VALSRC) $$(LDLIBS)
endef

$(eval $(call build_program,conv_linear_double,CONV_LINEAR,))
$(eval $(call build_program,conv_linear_float,CONV_LINEAR,-DUSE_FLOAT))
$(eval $(call build_program,conv_malloc_double,CONV_MALLOC,))
$(eval $(call build_program,conv_malloc_float,CONV_MALLOC,-DUSE_FLOAT))
$(eval $(call build_program,gemm_double,GEMM_BLOCKED,))
$(eval $(call build_program,gemm_float,GEMM_BLOCKED,-DUSE_FLOAT))

# Compara os kernels com a implementação ingênua (vários N, blocos, passos e kernels).
validate: $(addprefix $(BIN)/,$(VALIDATORS))
	@set -e; for test in $^; do ./$$test; done

# Relatório de vetorização dos laços dos kernels, para documentar no texto.
# Linhas "optimized" indicam laços vetorizados; "missed", os que não foram.
vecreport: | $(BIN)
	@: > $(BIN)/vecreport.txt
	@for spec in "conv_linear_double CONV_LINEAR" "conv_linear_float CONV_LINEAR -DUSE_FLOAT" \
	             "conv_malloc_double CONV_MALLOC" "conv_malloc_float CONV_MALLOC -DUSE_FLOAT" \
	             "gemm_double GEMM_BLOCKED" "gemm_float GEMM_BLOCKED -DUSE_FLOAT"; do \
	  set -- $$spec; name=$$1; op=$$2; shift 2; \
	  echo "== $$name" >> $(BIN)/vecreport.txt; \
	  $(CC) $(CFLAGS) -DOPERATION=$$op $$* -fopt-info-vec-optimized -c $(SRC) -o /dev/null 2>&1 \
	    | grep 'matrix_bench.c' | sort -u >> $(BIN)/vecreport.txt; \
	done
	@cat $(BIN)/vecreport.txt

# Registra o compilador e o alvo efetivo de -march (usado pelo run_benchmark.sh).
compiler-info:
	@$(CC) --version | head -n1
	@echo "CFLAGS=$(CFLAGS)"
	@$(CC) -march=$(MARCH) -Q --help=target 2>/dev/null | grep -E '^\s+-march=|^\s+-mtune=' || true

plots:
	python3 scripts/generate_plots.py --data $(DATA) --output $(FIGURES) --clean

clean:
	rm -rf $(BIN)
