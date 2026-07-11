#pragma once

#ifdef MATRIX_THREADED
#include<Windows.h>
#endif
#include<immintrin.h>

#include<stdio.h>
#include<stdint.h>
#include<stdlib.h>
#include<assert.h>
#include<math.h>


#ifdef FLOAT
#define real float
#define simd128 __m128
#define simd_setzero128 _mm_setzero_ps
#define simd_set128(d, c, b, a) _mm_set_ps(d, c, b, a)
#define simd_loadu128 _mm_loadu_ps
#define simd_add128 _mm_add_ps
#define simd_mul128 _mm_mul_ps
#define VALUE_COUNT_IN_ONE_REGISTER 4

#else
#define real double
#define simd128 __m128d
#define simd_setzero128 _mm_setzero_pd
#define simd_set128(d, c, b, a) _mm_set_pd(b, a)
#define simd_loadu128 _mm_loadu_pd
#define simd_add128 _mm_add_pd
#define simd_mul128 _mm_mul_pd
#define VALUE_COUNT_IN_ONE_REGISTER 2
#endif

typedef struct
{
    size_t rows;
    size_t cols;

    real *arr;
}mat;

//#define mat_print_n(m) mat_print(m);printf("\n"); 

mat mat_alloc(size_t rows, size_t cols);
void mat_print(mat m, const char *name);
void mat_sub_print(mat m, size_t rows, size_t cols, const char *name);
void mat_add(mat dest, mat a, mat b);
void mat_subt(mat dest, mat a, mat b);

void mat_mul(mat dest, mat a, mat b, int threads);

void mat_mul_scalar(mat dest, mat a, real s);
void mat_fill(mat m, real value);
void mat_rand(mat m, real low, real high);
void mat_trans(mat dest, mat src);
real mat_norm(mat m);

typedef struct mat_loop
{
    mat matrices[3];
    int range[2];

    int thread;
}mat_loop;


#define MAT_AT(mat, i, j) (mat).arr[(i)*(mat).cols + (j)]
#define MAT_PRINT(m) mat_print(m, #m) // '#' turns the name into a string
#define MAT_SUB_PRINT(m, row, col) mat_sub_print(m, row, col, #m)

#define MAX_THREADS 10

mat mat_alloc(size_t rows, size_t cols)
{
    mat m;

    m.rows = rows;
    m.cols = cols;

    m.arr = malloc(sizeof(*m.arr)*m.rows*m.cols);

    return m;
}

void mat_print(mat m, const char *name)
{
    printf("%s = [\n", name);
    for(size_t i = 0; i < m.rows; i++)
    {
        for(size_t j = 0; j < m.cols; j++)
        {
            printf("  %lf ", MAT_AT(m, i, j));
        }
        printf("\n");
    }
    printf("]\n");
}

void mat_sub_print(mat m, size_t rows, size_t cols, const char *name)
{
    printf("%s = [\n", name);
    for(size_t i = 0; i < rows; i++)
    {
        for(size_t j = 0; j < cols; j++)
        {
            printf("  %lf ", MAT_AT(m, i, j));
        }
        printf("\n");
    }
    printf("]\n");
}

void mat_add(mat dest, mat a, mat b)
{
    assert(a.rows == b.rows);
    assert(a.cols == b.cols);
    assert(dest.rows == a.rows);
    assert(dest.cols == a.cols);

    for(size_t i = 0; i < dest.rows; i++)
    {
        for(size_t j = 0; j < dest.cols; j++)
        {
            MAT_AT(dest, i, j) = MAT_AT(a, i, j) + MAT_AT(b, i, j);
        }
    }
}

void mat_subt(mat dest, mat a, mat b)
{
    assert(a.rows == b.rows);
    assert(a.cols == b.cols);

    for(size_t i = 0; i < dest.rows; i++)
    {
        for(size_t j = 0; j < dest.cols; j++)
        {
            MAT_AT(dest, i, j) = MAT_AT(a, i, j) - MAT_AT(b, i, j);
        }
    }
}


#ifdef MATRIX_THREADED
DWORD matmul_loop_thread(LPVOID matrices)
{
    mat_loop values = *(mat_loop *)matrices;

    mat dest = values.matrices[0];
    mat a = values.matrices[1];
    mat b = values.matrices[2];

    size_t low = values.range[0];
    size_t high = values.range[1];

    for (size_t i = low; i < high; i++)
    {
        for (size_t j = 0; j < dest.cols; j++)
        {
            simd128 acc = simd_setzero128(); 

            size_t k = 0;
            for (; k + (VALUE_COUNT_IN_ONE_REGISTER-1) < a.cols; k += VALUE_COUNT_IN_ONE_REGISTER)
            {
                if ((k + (VALUE_COUNT_IN_ONE_REGISTER-1)) >= a.cols || j >= b.rows) {
                    printf("Out of bounds array access.\n");
                    exit(1);
                }
                simd128 first  = simd_loadu128(&MAT_AT(a, i, k));
                simd128 second = simd_set128(MAT_AT(b, k+3, j), MAT_AT(b, k+2, j), MAT_AT(b, k+1, j), MAT_AT(b, k, j));

                acc = simd_add128(acc, simd_mul128(first, second));
            }

            real dot_prod = 0;
            for(int index = 0; index < VALUE_COUNT_IN_ONE_REGISTER; index++)
            {
                dot_prod += ((real *)&acc)[index];
            }

            for (; k < a.cols; k++)
            {
                dot_prod += MAT_AT(a, i, k) * MAT_AT(b, k, j);
            }

            MAT_AT(dest, i, j) = dot_prod;
        }
    }
}
#endif

void mat_mul(mat dest, mat a, mat b, int threads)
{

    assert(a.cols == b.rows);
    assert(dest.rows == a.rows);
    assert(dest.cols == b.cols);

#ifdef MATRIX_THREADED
    if(threads < 1)
    {
        threads = 1;
        fprintf(stderr, "Thread count less than 1\n");
    }
    else if(threads > MAX_THREADS) 
    {
        threads = MAX_THREADS;
        fprintf(stderr, "Thread count greater than MAX_THREADS\n");
    }

    mat_loop values = {{dest, a, b}, {0, dest.rows/2}, 1};
    mat_loop *datas[MAX_THREADS] = {0};
    for(int i = 0; i < threads; i++)
    {
        datas[i] = HeapAlloc(GetProcessHeap(), HEAP_ZERO_MEMORY, sizeof(*datas[i]));
        datas[i]->matrices[0] = dest;
        datas[i]->matrices[1] = a;
        datas[i]->matrices[2] = b;

        datas[i]->range[0] = (dest.rows / threads) * i;
        datas[i]->range[1] = ((dest.rows / threads) * (i+1));

        datas[i]->thread = i+1;
    }


    //printf("LOW: %lf, HIGH: %zu\n", data->range[0], data->range[1]);
    HANDLE handles[MAX_THREADS] = {0};
    for(int i = 0; i < threads; i++)
    {
        handles[i] = CreateThread(NULL, 0, matmul_loop_thread, datas[i], 0, NULL);
    }
    
    WaitForMultipleObjects(threads, handles, TRUE, INFINITE);

    for(int i = 0; i < threads; i++)
    {
        HeapFree(GetProcessHeap(), 0, datas[i]);
    }
    
#else
    for(size_t i = 0; i < dest.rows; i++)
    {
        for(size_t j = 0; j < dest.cols; j++)
        {
            simd128 acc = simd_setzero128(); 

            size_t k = 0;
            for (; k + (VALUE_COUNT_IN_ONE_REGISTER-1) < a.cols; k += VALUE_COUNT_IN_ONE_REGISTER)
            {
                if ((k + (VALUE_COUNT_IN_ONE_REGISTER-1)) >= a.cols || j >= b.rows) {
                    printf("Out of bounds array access.\n");
                    exit(1);
                }
                simd128 first  = simd_loadu128(&MAT_AT(a, i, k));
                simd128 second = simd_set128(MAT_AT(b, k+3, j), MAT_AT(b, k+2, j), MAT_AT(b, k+1, j), MAT_AT(b, k, j));

                acc = simd_add128(acc, simd_mul128(first, second));
            }

            real dot_prod = 0;
            for(int index = 0; index < VALUE_COUNT_IN_ONE_REGISTER; index++)
            {
                dot_prod += ((real *)&acc)[index];
            }

            for (; k < a.cols; k++)
            {
                dot_prod += MAT_AT(a, i, k) * MAT_AT(b, k, j);
            }

            MAT_AT(dest, i, j) = dot_prod;
        }
    }
#endif
    

}

void mat_mul_scalar(mat dest, mat a, real s)
{
    assert(dest.rows == a.rows);
    assert(dest.cols == a.cols);
    for(size_t i = 0; i < dest.rows * dest.cols; i++)
    {
        *dest.arr++ = s * (*a.arr++);
    }
}

void mat_fill(mat m, real value)
{
    for(size_t i = 0; i < m.rows * m.cols; i++)
    {    
        *m.arr++ = value;   
    }
}

void mat_rand(mat m, real low, real high)
{
    for(size_t i = 0; i < m.rows; i++)
    {
        for(size_t j = 0; j < m.cols; j++)
        {
            MAT_AT(m, i, j) = (rand()/(real)RAND_MAX) * (high - low) + low;
        }
    }
}

void mat_trans(mat dest, mat src)
{
    assert(dest.rows == src.cols);
    assert(dest.cols == src.rows);

    for(size_t i = 0; i < src.rows; i++)
    {
        for(size_t j = 0; j < src.cols; j++)
        {
            MAT_AT(dest, j, i) = MAT_AT(src, i, j);
        }
    }
}

real mat_norm(mat m)
{
    real result = 0.f;
    for(size_t i = 0; i < m.rows; i++)
    {
        for(size_t j = 0; j < m.cols; j++)
        {
            result += MAT_AT(m, i, j) * MAT_AT(m, i, j);
        }
    }
    return sqrtf(result);
}