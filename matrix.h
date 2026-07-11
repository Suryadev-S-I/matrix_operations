#pragma once

#ifdef MATRIX_THREADED
#include<Windows.h>
#endif

#include<stdio.h>
#include<stdint.h>
#include<stdlib.h>
#include<assert.h>
#include<math.h>


#ifdef __cplusplus
extern "C" {
#endif

#ifdef FLOAT
#define real float
#else
#define real double
#endif

typedef struct
{
    size_t rows;
    size_t cols;

    real *arr;
}mat;

typedef struct mat_loop
{
    mat matrices[3];
    int range[2];

    int thread;
}mat_loop;

mat mat_alloc(size_t rows, size_t cols);
void mat_fill(mat m, real value);
void mat_rand(mat m, real low, real high);

void mat_trans(mat dest, mat src);
void mat_add(mat dest, mat a, mat b);
void mat_subt(mat dest, mat a, mat b);
void mat_mul(mat dest, mat a, mat b, int threads);
void mat_mul_scalar(mat dest, mat a, real s);
real mat_norm(mat m);

void mat_print(mat m, const char *name);
void mat_sub_print(mat m, size_t rows, size_t cols, const char *name);



#define MAT_AT(mat, i, j) (mat).arr[(i)*(mat).cols + (j)]
#define MAT_PRINT(m) mat_print(m, #m) // '#' turns the name into a string
#define MAT_SUB_PRINT(m, row, col) mat_sub_print(m, row, col, #m)

#ifndef MAX_THREADS
#define MAX_THREADS 8
#endif

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

    for(size_t i = low; i < high; i++)
    {
        for(size_t j = 0; j < dest.cols; j++)
        {
            real dot_prod = 0;
            for(size_t k = 0; k < a.cols; k++)
            {
                dot_prod += MAT_AT(a, i, k) * MAT_AT(b, k, j);
            }
            MAT_AT(dest, i, j) = dot_prod;
        }
    }

    return 1;
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

    mat_loop *datas[MAX_THREADS] = {0};
    HANDLE heap_handle = GetProcessHeap();
    for(int i = 0; i < threads; i++)
    {
        datas[i] = HeapAlloc(heap_handle, HEAP_ZERO_MEMORY, sizeof(*datas[i]));
        datas[i]->matrices[0] = dest;
        datas[i]->matrices[1] = a;
        datas[i]->matrices[2] = b;

        datas[i]->range[0] = (dest.rows / threads) * i;
        datas[i]->range[1] = ((dest.rows / threads) * (i+1));

        datas[i]->thread = i+1;
    }

    HANDLE handles[MAX_THREADS] = {0};
    
    for(int i = 0; i < threads; i++)
    {
        handles[i] = CreateThread(NULL, 0, matmul_loop_thread, datas[i], 0, NULL);
    }
    
    WaitForMultipleObjects(threads, handles, TRUE, INFINITE);

    for(int i = 0; i < threads; i++)
    {
        HeapFree(heap_handle, 0, datas[i]);
    }
#else

    for(size_t i = 0; i < dest.rows; i++)
    {
        for(size_t j = 0; j < dest.cols; j++)
        {
            real dot_prod = 0;
            for(size_t k = 0; k < a.cols; k++)
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

#ifdef __cplusplus
}
#endif