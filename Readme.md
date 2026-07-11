## USAGE
* Include matrix.h for non-simd operations.
* Incldue matrix_simd.h for simd, x86 operations.
* Define 'real' as double to use doubles instead of floats.
* Define MATRIX_THREADED to turn on multithreadiing, **WINDOWS ONLY**.
* Define MAX_THREADS to change the maximum number of threads, default 8.
