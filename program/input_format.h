#ifndef TCC2_INPUT_FORMAT_H
#define TCC2_INPUT_FORMAT_H

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { INPUT_MATRIX = 1, INPUT_GEMM = 2, INPUT_KERNEL = 3 };
enum { INPUT_TYPE_FLOAT = 1, INPUT_TYPE_DOUBLE = 2 };

typedef struct {
    char magic[8];
    uint32_t version;
    uint32_t kind;
    uint32_t n;
    int32_t dist;
    uint32_t seed;
    uint32_t data_type;
    uint64_t count;
} input_header_t;

typedef struct {
    input_header_t header;
    void *values;
} input_data_t;

_Static_assert(sizeof(input_header_t) == 40, "cabecalho de entrada deve ter 40 bytes");

static int input_load(const char *path, uint32_t expected_kind, uint32_t expected_type,
                      size_t element_size, input_data_t *input) {
    static const char expected_magic[8] = {'T','C','C','2','I','0','1','\0'};
    FILE *file = fopen(path, "rb");
    memset(input, 0, sizeof(*input));
    if (!file) {
        fprintf(stderr, "Erro: nao foi possivel abrir a entrada %s.\n", path);
        return 0;
    }
    if (fread(&input->header, sizeof(input->header), 1, file) != 1 ||
        memcmp(input->header.magic, expected_magic, sizeof(expected_magic)) != 0 ||
        input->header.version != 2 || input->header.kind != expected_kind ||
        input->header.data_type != expected_type ||
        input->header.n == 0 || input->header.count == 0 ||
        input->header.count > SIZE_MAX / element_size) {
        fprintf(stderr, "Erro: cabecalho invalido em %s.\n", path);
        fclose(file);
        return 0;
    }
    input->values = malloc((size_t)input->header.count * element_size);
    if (!input->values ||
        fread(input->values, element_size, (size_t)input->header.count, file) != input->header.count) {
        fprintf(stderr, "Erro: dados incompletos em %s.\n", path);
        free(input->values);
        input->values = NULL;
        fclose(file);
        return 0;
    }
    if (fgetc(file) != EOF) {
        fprintf(stderr, "Erro: dados excedentes em %s.\n", path);
        free(input->values);
        input->values = NULL;
        fclose(file);
        return 0;
    }
    fclose(file);
    return 1;
}

static void input_release(input_data_t *input) {
    free(input->values);
    input->values = NULL;
}

static int dump_as_double(const char *path, const void *values, uint64_t count, int is_float) {
    FILE *file = fopen(path, "wb");
    if (!file || fwrite(&count, sizeof(count), 1, file) != 1) {
        if (file) fclose(file);
        return 0;
    }
    for (uint64_t i = 0; i < count; i++) {
        double value = is_float ? (double)((const float *)values)[i]
                                : ((const double *)values)[i];
        if (fwrite(&value, sizeof(value), 1, file) != 1) {
            fclose(file);
            return 0;
        }
    }
    return fclose(file) == 0;
}

#endif
