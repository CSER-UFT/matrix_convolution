/*
 * matrix_bench.c: convolução 2D (alocação contígua ou por linhas) e GEMM blocada,
 * em versão completa (--step 1) ou aproximada por perfuração de laço (--step S > 1).
 *
 * Mudanças em relação à versão da campanha de 22/09:
 *   1. O tempo é medido dentro do programa, somente em volta do kernel, com
 *      clock_gettime(CLOCK_MONOTONIC). Leitura da entrada, alocação, checksum e
 *      gravação ficam fora da medição.
 *   2. Os contadores do perf (inclusive energia RAPL) cobrem o mesmo lote de
 *      repetições do kernel: o programa liga a contagem antes da primeira
 *      repetição medida e desliga depois da última, pelo modo --control do
 *      perf stat (ver run_benchmark.sh).
 *   3. O kernel é executado várias vezes (aquecimento descartado) e o programa
 *      informa mediana, mínimo e média; os contadores são divididos por Reps no
 *      parse_results.py.
 *   4. Os laços foram reorganizados para que o laço mais interno seja contíguo e
 *      sem redução (vetorizável sem -ffast-math):
 *        GEMM: ordem i, k, j dentro dos blocos;
 *        convolução: o laço interno percorre j (colunas da saída).
 *      A ordem de acumulação de cada elemento da saída é a mesma da versão
 *      anterior, então o arredondamento da versão completa não muda.
 *   5. O grau de aproximação virou parâmetro de execução (--step). A versão
 *      completa e as aproximadas usam o mesmo binário e o mesmo código.
 *   6. skip_k: o fator de escala é aplicado uma vez por a[i][k], fora do laço
 *      interno. Os índices k mantidos são os múltiplos de step, independentes
 *      do tamanho do bloco.
 *   7. skip_kernel: os pesos mantidos são reescalados para preservar a norma L1
 *      do kernel. Para kernels positivos isso é idêntico à renormalização
 *      anterior (soma 1); para kernels de soma zero (Sobel) a técnica deixa de
 *      falhar.
 *
 * Uso:
 *   conv: <prog> <matriz.bin> <kernel.bin> [opções]
 *   gemm: <prog> <entrada_gemm.bin> <bloco> [opções]
 * Opções:
 *   --step S        passo da perfuração (1 = versão completa; padrão 1)
 *   --reps R        número fixo de repetições medidas (desliga o modo adaptativo)
 *   --min-time T    modo adaptativo: repete até somar T segundos (padrão 0.2)
 *   --min-reps R    modo adaptativo: mínimo de repetições (padrão 3)
 *   --max-reps R    modo adaptativo: máximo de repetições (padrão 1000)
 *   --dump arquivo  grava a saída (em double) para o cálculo de erro
 */
#include <errno.h>
#include <inttypes.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "input_format.h"

#define CONV_LINEAR 1
#define CONV_MALLOC 2
#define GEMM_BLOCKED 3

#ifndef OPERATION
#error OPERATION deve ser CONV_LINEAR, CONV_MALLOC ou GEMM_BLOCKED
#endif
#ifndef PROGRAM_NAME
#define PROGRAM_NAME "matrix_bench"
#endif

#ifdef USE_FLOAT
typedef float real_t;
#define PRECISION_NAME "float"
#define IS_FLOAT 1
#define INPUT_DATA_TYPE INPUT_TYPE_FLOAT
#else
typedef double real_t;
#define PRECISION_NAME "double"
#define IS_FLOAT 0
#define INPUT_DATA_TYPE INPUT_TYPE_DOUBLE
#endif

#if OPERATION == GEMM_BLOCKED
#define OPERATION_NAME "gemm"
#define APPROX_NAME "skip_k"
#elif OPERATION == CONV_LINEAR
#define OPERATION_NAME "conv_linear"
#define APPROX_NAME "skip_kernel"
#elif OPERATION == CONV_MALLOC
#define OPERATION_NAME "conv_malloc"
#define APPROX_NAME "skip_kernel"
#else
#error OPERATION invalida
#endif

#if OPERATION != GEMM_BLOCKED
static const char *const KERNEL_TYPE_NAMES[] = {"rand", "gauss", "sobel"};
#endif

static inline int imin(int a, int b) { return a < b ? a : b; }

/* Quantidade de índices em [0, extent) que são múltiplos de step. */
static int kept_count(int extent, int step) { return (extent + step - 1) / step; }

/* ------------------------------------------------------------------------- */
/* Kernels                                                                     */
/* ------------------------------------------------------------------------- */

#if OPERATION != GEMM_BLOCKED
/*
 * Prepara o kernel da convolução para o passo step: zera os pesos descartados e
 * reescala os mantidos para preservar a norma L1. Executado fora da medição.
 */
static int prepare_kernel(real_t *w, int k, int step) {
    double l1_full = 0.0, l1_kept = 0.0;
    if (step == 1) return 1;
    for (int i = 0; i < k; i++)
        for (int j = 0; j < k; j++) {
            double v = fabs((double)w[i * k + j]);
            l1_full += v;
            if (i % step == 0 && j % step == 0) l1_kept += v;
        }
    if (l1_kept == 0.0) return 0;
    const real_t factor = (real_t)(l1_full / l1_kept);
    for (int i = 0; i < k; i++)
        for (int j = 0; j < k; j++) {
            if (i % step == 0 && j % step == 0) w[i * k + j] *= factor;
            else w[i * k + j] = (real_t)0;
        }
    return 1;
}
#endif

#if OPERATION == CONV_LINEAR
/* Convolução "válida": a borda de largura k/2 da saída não é escrita (fica 0). */
static void conv_linear(real_t *restrict out, const real_t *restrict in, int n,
                        const real_t *restrict w, int k, int step) {
    const int off = k / 2, width = n - 2 * off;
    for (int i = off; i < n - off; i++) {
        real_t *restrict dst = out + (size_t)i * n + off;
        for (int j = 0; j < width; j++) dst[j] = (real_t)0;
        for (int ki = 0; ki < k; ki += step) {
            const real_t *restrict row = in + (size_t)(i - off + ki) * n;
            for (int kj = 0; kj < k; kj += step) {
                const real_t wk = w[ki * k + kj];
                const real_t *restrict src = row + kj;
                for (int j = 0; j < width; j++) dst[j] += wk * src[j];
            }
        }
    }
}
#elif OPERATION == CONV_MALLOC
static real_t **allocate_rows(int rows, int cols, int zero) {
    real_t **matrix = calloc((size_t)rows, sizeof(*matrix));
    if (!matrix) return NULL;
    for (int i = 0; i < rows; i++) {
        matrix[i] = zero ? calloc((size_t)cols, sizeof(real_t))
                         : malloc((size_t)cols * sizeof(real_t));
        if (!matrix[i]) {
            for (int j = 0; j < i; j++) free(matrix[j]);
            free(matrix);
            return NULL;
        }
    }
    return matrix;
}

static void release_rows(real_t **matrix, int rows) {
    if (!matrix) return;
    for (int i = 0; i < rows; i++) free(matrix[i]);
    free(matrix);
}

static void conv_rows(real_t *const *out, real_t *const *in, int n,
                      real_t *const *w, int k, int step) {
    const int off = k / 2, width = n - 2 * off;
    for (int i = off; i < n - off; i++) {
        real_t *restrict dst = out[i] + off;
        for (int j = 0; j < width; j++) dst[j] = (real_t)0;
        for (int ki = 0; ki < k; ki += step) {
            const real_t *restrict row = in[i - off + ki];
            for (int kj = 0; kj < k; kj += step) {
                const real_t wk = w[ki][kj];
                const real_t *restrict src = row + kj;
                for (int j = 0; j < width; j++) dst[j] += wk * src[j];
            }
        }
    }
}
#else
/*
 * C = scale * soma_{k múltiplo de step} A[:,k] B[k,:]
 * Ordem ii, kk, jj nos blocos e i, k, j dentro do bloco: o laço interno percorre
 * uma linha de B e uma linha de C de forma contígua.
 */
static void gemm_blocked(real_t *restrict c, const real_t *restrict a, const real_t *restrict b,
                         int n, int bs, int step, real_t scale) {
    memset(c, 0, (size_t)n * (size_t)n * sizeof(real_t));
    for (int ii = 0; ii < n; ii += bs) {
        const int imax = imin(ii + bs, n);
        for (int kk = 0; kk < n; kk += bs) {
            const int kmax = imin(kk + bs, n);
            const int k0 = ((kk + step - 1) / step) * step;
            for (int jj = 0; jj < n; jj += bs) {
                const int jmax = imin(jj + bs, n);
                for (int i = ii; i < imax; i++) {
                    real_t *restrict crow = c + (size_t)i * n;
                    const real_t *restrict arow = a + (size_t)i * n;
                    for (int k = k0; k < kmax; k += step) {
                        const real_t aik = arow[k] * scale;
                        const real_t *restrict brow = b + (size_t)k * n;
                        for (int j = jj; j < jmax; j++) crow[j] += aik * brow[j];
                    }
                }
            }
        }
    }
}
#endif

/* ------------------------------------------------------------------------- */
/* Medição                                                                     */
/* ------------------------------------------------------------------------- */

typedef struct {
    int step;
    int reps;       /* > 0: número fixo de repetições medidas */
    int min_reps;
    int max_reps;
    double min_time;
    const char *dump;
} options_t;

typedef struct {
    double median, min, mean;
    int reps;
} timing_t;

typedef void (*kernel_fn)(void *ctx);

static int perf_ctl_fd = -1, perf_ack_fd = -1;

/* O run_benchmark.sh informa os descritores do --control do perf por variáveis de ambiente. */
static void perf_setup(void) {
    const char *ctl = getenv("TCC2_PERF_CTL_FD");
    const char *ack = getenv("TCC2_PERF_ACK_FD");
    if (ctl && ack && *ctl && *ack) {
        perf_ctl_fd = atoi(ctl);
        perf_ack_fd = atoi(ack);
    }
}

static int perf_command(const char *command) {
    char ack[32];
    size_t length;
    if (perf_ctl_fd < 0) return 1; /* execução fora do perf */
    length = strlen(command);
    if (write(perf_ctl_fd, command, length) != (ssize_t)length) return 0;
    return read(perf_ack_fd, ack, sizeof ack) > 0;
}

static double now_sec(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + (double)t.tv_nsec * 1e-9;
}

static int compare_double(const void *x, const void *y) {
    double a = *(const double *)x, b = *(const double *)y;
    return (a > b) - (a < b);
}

static int run_timed(kernel_fn fn, void *ctx, const options_t *opt, timing_t *result) {
    const int capacity = opt->reps > 0 ? opt->reps : opt->max_reps;
    double *times = malloc((size_t)capacity * sizeof(*times));
    double total = 0.0;
    int r = 0;
    if (!times) return 0;
    fn(ctx); /* aquecimento: faltas de página, caches e preditores; não medido */
    /*
     * Uma única janela do perf para todo o lote de repetições. Isso evita a
     * latência do protocolo --control a cada repetição e é necessário para a
     * energia: o contador RAPL só é atualizado a cada ~1 ms, então janelas
     * curtas teriam erro de quantização grande. Entre as repetições só há duas
     * chamadas de clock_gettime (vDSO), custo desprezível frente ao kernel.
     */
    if (!perf_command("enable\n")) { free(times); return 0; }
    while (r < capacity) {
        if (opt->reps == 0 && r >= opt->min_reps && total >= opt->min_time) break;
        const double t0 = now_sec();
        fn(ctx);
        const double t1 = now_sec();
        times[r++] = t1 - t0;
        total += t1 - t0;
    }
    if (!perf_command("disable\n")) { free(times); return 0; }
    qsort(times, (size_t)r, sizeof(*times), compare_double);
    result->reps = r;
    result->min = times[0];
    result->median = (r % 2) ? times[r / 2] : 0.5 * (times[r / 2 - 1] + times[r / 2]);
    result->mean = total / r;
    free(times);
    return 1;
}

static int parse_int(const char *text, int *value) {
    char *end = NULL;
    long parsed;
    errno = 0;
    parsed = strtol(text, &end, 10);
    if (errno || end == text || *end || parsed <= 0 || parsed > INT_MAX) return 0;
    *value = (int)parsed;
    return 1;
}

static int parse_double(const char *text, double *value) {
    char *end = NULL;
    errno = 0;
    *value = strtod(text, &end);
    return !errno && end != text && !*end && *value >= 0.0;
}

static int parse_options(int argc, char **argv, int first, options_t *opt) {
    opt->step = 1; opt->reps = 0; opt->min_reps = 3; opt->max_reps = 1000;
    opt->min_time = 0.2; opt->dump = NULL;
    for (int i = first; i < argc; i++) {
        if (i + 1 >= argc) return 0;
        const char *name = argv[i], *value = argv[++i];
        int ok;
        if (!strcmp(name, "--step")) ok = parse_int(value, &opt->step);
        else if (!strcmp(name, "--reps")) ok = parse_int(value, &opt->reps);
        else if (!strcmp(name, "--min-reps")) ok = parse_int(value, &opt->min_reps);
        else if (!strcmp(name, "--max-reps")) ok = parse_int(value, &opt->max_reps);
        else if (!strcmp(name, "--min-time")) ok = parse_double(value, &opt->min_time);
        else if (!strcmp(name, "--dump")) { opt->dump = value; ok = 1; }
        else ok = 0;
        if (!ok) return 0;
    }
    return opt->min_reps <= opt->max_reps;
}

static void report(const input_data_t *input, int parameter, const char *kernel_type,
                   const options_t *opt, const timing_t *t, long kept, double checksum) {
    printf("Program=%s Operation=%s Precision=%s Technique=%s Step=%d N=%u Dist=%d "
           "Parameter=%d Kernel=%s Seed=%u\n",
           PROGRAM_NAME, OPERATION_NAME, PRECISION_NAME,
           opt->step == 1 ? "full" : APPROX_NAME, opt->step, input->header.n,
           input->header.dist, parameter, kernel_type, input->header.seed);
    printf("Reps %d\n", t->reps);
    printf("TimeMedian %.9e\n", t->median);
    printf("TimeMin %.9e\n", t->min);
    printf("TimeMean %.9e\n", t->mean);
    printf("KeptTerms %ld\n", kept);
    printf("Checksum %.17g\n", checksum);
}

/* ------------------------------------------------------------------------- */
/* Programa principal                                                          */
/* ------------------------------------------------------------------------- */

#if OPERATION == GEMM_BLOCKED
typedef struct { real_t *c; const real_t *a, *b; int n, bs, step; real_t scale; } gemm_ctx_t;
static void gemm_call(void *p) {
    gemm_ctx_t *x = p;
    gemm_blocked(x->c, x->a, x->b, x->n, x->bs, x->step, x->scale);
}
#elif OPERATION == CONV_LINEAR
typedef struct { real_t *out; const real_t *in, *w; int n, k, step; } conv_ctx_t;
static void conv_call(void *p) {
    conv_ctx_t *x = p;
    conv_linear(x->out, x->in, x->n, x->w, x->k, x->step);
}
#else
typedef struct { real_t **out, **in, **w; int n, k, step; } conv_ctx_t;
static void conv_call(void *p) {
    conv_ctx_t *x = p;
    conv_rows(x->out, x->in, x->n, x->w, x->k, x->step);
}
#endif

int main(int argc, char **argv) {
    input_data_t input = {0}, kernel_input = {0};
    options_t opt;
    timing_t timing;
    int ok = 0;

    perf_setup();

#if OPERATION == GEMM_BLOCKED
    int block = 0;
    if (argc < 3 || !parse_int(argv[2], &block) || !parse_options(argc, argv, 3, &opt)) {
        fprintf(stderr, "Uso: %s <entrada_gemm.bin> <bloco> [--step S] [--reps R | --min-time T] [--dump arq]\n", argv[0]);
        return 1;
    }
    if (!input_load(argv[1], INPUT_GEMM, INPUT_DATA_TYPE, sizeof(real_t), &input)) return 1;
    const int n = (int)input.header.n;
    const uint64_t count = (uint64_t)n * (uint64_t)n;
    if (input.header.count != 2 * count || opt.step > n) {
        fprintf(stderr, "Erro: entrada GEMM ou passo incompatível.\n");
        goto cleanup;
    }
    const real_t *values = input.values;
    real_t *c = calloc((size_t)count, sizeof(real_t));
    if (!c) { fprintf(stderr, "Erro: falha de alocação.\n"); goto cleanup; }
    const int kept = kept_count(n, opt.step);
    gemm_ctx_t ctx = {c, values, values + count, n, block, opt.step, (real_t)((double)n / (double)kept)};
    if (!run_timed(gemm_call, &ctx, &opt, &timing)) {
        fprintf(stderr, "Erro: falha na medição ou no controle do perf.\n");
        free(c); goto cleanup;
    }
    double checksum = 0.0;
    for (uint64_t i = 0; i < count; i++) checksum += (double)c[i];
    if (!isfinite(checksum)) { fprintf(stderr, "Erro: resultado não finito.\n"); free(c); goto cleanup; }
    if (opt.dump && !dump_as_double(opt.dump, c, count, IS_FLOAT)) {
        fprintf(stderr, "Erro: não foi possível gravar %s.\n", opt.dump); free(c); goto cleanup;
    }
    report(&input, block, "none", &opt, &timing, kept, checksum);
    free(c);
    ok = 1;
#else
    if (argc < 3 || !parse_options(argc, argv, 3, &opt)) {
        fprintf(stderr, "Uso: %s <matriz.bin> <kernel.bin> [--step S] [--reps R | --min-time T] [--dump arq]\n", argv[0]);
        return 1;
    }
    if (!input_load(argv[1], INPUT_MATRIX, INPUT_DATA_TYPE, sizeof(real_t), &input) ||
        !input_load(argv[2], INPUT_KERNEL, INPUT_DATA_TYPE, sizeof(real_t), &kernel_input)) goto cleanup;
    const int n = (int)input.header.n;
    const int k = (int)kernel_input.header.n;
    const uint64_t count = (uint64_t)n * (uint64_t)n;
    const int type_id = kernel_input.header.dist;
    const char *kernel_type = (type_id >= 0 && type_id < 3) ? KERNEL_TYPE_NAMES[type_id] : "unknown";
    if (input.header.count != count || kernel_input.header.count != (uint64_t)k * (uint64_t)k ||
        !(k & 1) || k > n || opt.step > k) {
        fprintf(stderr, "Erro: matriz, kernel ou passo incompatível.\n");
        goto cleanup;
    }
    real_t *w = malloc((size_t)k * (size_t)k * sizeof(real_t));
    if (!w) { fprintf(stderr, "Erro: falha de alocação.\n"); goto cleanup; }
    memcpy(w, kernel_input.values, (size_t)k * (size_t)k * sizeof(real_t));
    if (!prepare_kernel(w, k, opt.step)) {
        fprintf(stderr, "Erro: pesos mantidos do kernel têm norma zero.\n"); free(w); goto cleanup;
    }
    const long kept = (long)kept_count(k, opt.step) * kept_count(k, opt.step);
    double checksum = 0.0;
#if OPERATION == CONV_LINEAR
    real_t *out = calloc((size_t)count, sizeof(real_t));
    if (!out) { fprintf(stderr, "Erro: falha de alocação.\n"); free(w); goto cleanup; }
    conv_ctx_t ctx = {out, input.values, w, n, k, opt.step};
    if (!run_timed(conv_call, &ctx, &opt, &timing)) {
        fprintf(stderr, "Erro: falha na medição ou no controle do perf.\n"); free(out); free(w); goto cleanup;
    }
    for (uint64_t i = 0; i < count; i++) checksum += (double)out[i];
    if (!isfinite(checksum) || (opt.dump && !dump_as_double(opt.dump, out, count, IS_FLOAT))) {
        fprintf(stderr, "Erro: resultado não finito ou falha ao gravar a saída.\n"); free(out); free(w); goto cleanup;
    }
    free(out);
#else
    const real_t *values = input.values;
    real_t **matrix = allocate_rows(n, n, 0);
    real_t **kernel = allocate_rows(k, k, 0);
    real_t **out = allocate_rows(n, n, 1);
    real_t *flat = NULL;
    int fine = matrix && kernel && out;
    if (fine) {
        for (int i = 0; i < n; i++) memcpy(matrix[i], values + (size_t)i * n, (size_t)n * sizeof(real_t));
        for (int i = 0; i < k; i++) memcpy(kernel[i], w + (size_t)i * k, (size_t)k * sizeof(real_t));
        conv_ctx_t ctx = {out, matrix, kernel, n, k, opt.step};
        fine = run_timed(conv_call, &ctx, &opt, &timing);
    }
    if (fine) {
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++) checksum += (double)out[i][j];
        fine = isfinite(checksum);
    }
    if (fine && opt.dump) {
        flat = malloc((size_t)count * sizeof(real_t));
        fine = flat != NULL;
        if (fine) {
            for (int i = 0; i < n; i++) memcpy(flat + (size_t)i * n, out[i], (size_t)n * sizeof(real_t));
            fine = dump_as_double(opt.dump, flat, count, IS_FLOAT);
        }
    }
    free(flat);
    release_rows(matrix, n); release_rows(kernel, k); release_rows(out, n);
    if (!fine) { fprintf(stderr, "Erro: falha de alocação, medição, resultado ou gravação.\n"); free(w); goto cleanup; }
#endif
    free(w);
    report(&input, k, kernel_type, &opt, &timing, kept, checksum);
    ok = 1;
#endif

cleanup:
    input_release(&kernel_input);
    input_release(&input);
    return ok ? 0 : 1;
}
