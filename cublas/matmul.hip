#include "operator.h"
#include <cstdlib>
#include <cstdio>
#include <vector>
#include <chrono>
#include <iostream>
#include <hip/hip_runtime.h>
#include <rocblas.h>
#include "utils.h"
// #include <cublasLt.h>

void float16(uint16_t *__restrict out, const float in) {
    uint32_t inu = *((uint32_t * ) & in);
    uint32_t t1;
    uint32_t t2;
    uint32_t t3;

    t1 = inu & 0x7fffffffu;                 // Non-sign bits
    t2 = inu & 0x80000000u;                 // Sign bit
    t3 = inu & 0x7f800000u;                 // Exponent

    t1 >>= 13u;                             // Align mantissa on MSB
    t2 >>= 16u;                             // Shift sign bit into position

    t1 -= 0x1c000;                         // Adjust bias

    t1 = (t3 < 0x38800000u) ? 0 : t1;       // Flush-to-zero
    t1 = (t3 > 0x8e000000u) ? 0x7bff : t1;  // Clamp-to-max
    t1 = (t3 == 0 ? 0 : t1);               // Denormals-as-zero

    t1 |= t2;                              // Re-insert sign bit

    *((uint16_t *) out) = t1;
};

void MatMul::performance_measuring(Simulator *simu, int rounds)
{
    if (config.args.size() < 8)
    {
        std::cerr << "invalid configuration of MatMul!\n";
        exit(0);
    }

    int M = config.args[0];
    int N = config.args[1];
    int K = config.args[2];
    int TRANSA = config.args[3];
    int TRANSB = config.args[4];
    int TENSOR_OP = config.args[5];
    int ALGO_ID = config.args[6];
    int dtype = config.args[7];

    size_t dsize = sizeof(float);
    if (dtype == DTYPE_HALF)
        dsize = sizeof(rocblas_half);

    float alpha = 1.0, beta = 0.0;

    rocblas_handle handle;
    CUBLAS_CALL(rocblas_create_handle(&handle));

    int max_size_a = M * K, max_size_b = K * N, max_size_c = M * N;
    void *devPtrA = 0, *devPtrB = 0, *devPtrC = 0;
    CUDA_CALL(hipMalloc((void **)&devPtrA, max_size_a * dsize));
    CUDA_CALL(hipMalloc((void **)&devPtrB, max_size_b * dsize));
    CUDA_CALL(hipMalloc((void **)&devPtrC, max_size_c * dsize));

    void *A = malloc(max_size_a * dsize);
    void *B = malloc(max_size_b * dsize);
    void *C = malloc(max_size_c * dsize);

    if (dtype == DTYPE_FLOAT)
    {
        for (int i = 0; i < max_size_a; i++)
            ((float *)A)[i] = rand() % 5;
        for (int i = 0; i < max_size_b; i++)
            ((float *)B)[i] = rand() % 5;
        for (int i = 0; i < max_size_c; i++)
            ((float *)C)[i] = rand() % 5;
    }else if (dtype == DTYPE_HALF)
    {
        for (int i = 0; i < max_size_a; i++)
            ((rocblas_half *)A)[i] = float(rand() % 5);
        for (int i = 0; i < max_size_b; i++)
            ((rocblas_half *)B)[i] = float(rand() % 5);
        for (int i = 0; i < max_size_c; i++)
            ((rocblas_half *)C)[i] = float(rand() % 5);
    }
    

    CUDA_CALL(hipMemcpy(devPtrA, A, max_size_a * dsize, hipMemcpyHostToDevice));
    CUDA_CALL(hipMemcpy(devPtrB, B, max_size_b * dsize, hipMemcpyHostToDevice));
    CUDA_CALL(hipMemcpy(devPtrC, C, max_size_c * dsize, hipMemcpyHostToDevice));

    rocblas_operation transa, transb;
    int lda, ldb, ldc = M;
    switch (TRANSA)
    {
    case 0:
        transa = rocblas_operation_none;
        lda = M;
        break;
    case 1:
        transa = rocblas_operation_transpose;
        lda = K;
        break;
    default:
        printf("Invalid transa\n");
        exit(1);
    }
    switch (TRANSB)
    {
    case 0:
        transb = rocblas_operation_none;
        ldb = K;
        break;
    case 1:
        transb = rocblas_operation_transpose;
        ldb = N;
    default:
        printf("Invalid transb\n");
        exit(1);
    }

    rocblas_gemm_algo algo;
    if (TENSOR_OP == 0)
    {
        switch (ALGO_ID)
        {
        case -1:
            algo = rocblas_gemm_algo_standard;
            break;
        default:
            printf("Invalid rocblas algo id\n");
            exit(1);
        }
    }
    else
    {
        switch (ALGO_ID)
        {
        case -1:
            algo = rocblas_gemm_algo_standard;
            break;
        default:
            printf("Invalid rocblas algo id\n");
            exit(1);
        }
    }

    double avg_time = 0;
    float time = 0;
    hipEvent_t start, stop;
    CUDA_CALL(hipEventCreate(&start));
    CUDA_CALL(hipEventCreate(&stop));
    
    // warmup
    for (int j = 0; j < 10; ++j){
        CUBLAS_CALL(rocblas_gemm_ex(handle, transa, transb, M, N, K, &alpha,
                                    devPtrA, rocblas_datatype_f16_r, lda,
                                    devPtrB, rocblas_datatype_f16_r, ldb,
                                    &beta, devPtrC, rocblas_datatype_f16_r, ldc, 
                                    devPtrC, rocblas_datatype_f16_r, ldc, rocblas_datatype_f32_r,
                                    algo, 0, 0));
    }

    for (int i = 0; i < rounds; ++i)
    {
        hipEventRecord(start);
        if (dtype == DTYPE_FLOAT)
        {
            printf("Invalid rocblas dtype\n");
            exit(1);        
        }
        else if (dtype == DTYPE_HALF)
        {
            CUBLAS_CALL(rocblas_gemm_ex(handle, transa, transb, M, N, K, &alpha,
                                     devPtrA, rocblas_datatype_f16_r, lda,
                                     devPtrB, rocblas_datatype_f16_r, ldb,
                                     &beta, devPtrC, rocblas_datatype_f16_r, ldc, 
                                     devPtrC, rocblas_datatype_f16_r, ldc, rocblas_datatype_f32_r,
                                     algo, 0, 0));
        }

        CUDA_CALL(hipEventRecord(stop));
        CUDA_CALL(hipEventSynchronize(stop));
        CUDA_CALL(hipEventElapsedTime(&time, start, stop));
        avg_time += time;
        std::cout<< "time: " << time << std::endl;
    }
    avg_time /= rounds;

    CUDA_CALL(hipFree(devPtrA));
    CUDA_CALL(hipFree(devPtrB));
    CUDA_CALL(hipFree(devPtrC));
    free(A);
    free(B);
    free(C);
    CUBLAS_CALL(rocblas_destroy_handle(handle));

    double tflops = (double)M * (double)N * (double)K * 2 / avg_time / 1000000000.0;

    simu->updatePfMap(config, avg_time, tflops, {});
}
