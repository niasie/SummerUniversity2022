#pragma once

#include <iostream>

#include <cstdlib>
#include <cublas_v2.h>

static inline void cuda_check_last_kernel(std::string const& errstr) {
    auto status = cudaGetLastError();
    if(status != cudaSuccess) {
        std::cout << "error: CUDA kernel launch :" << errstr << " : "
                  << cudaGetErrorString(status) << std::endl;
        exit(-1);
    }
}

static inline void cuda_check_status(cudaError_t error_code) {
    if(error_code != cudaSuccess) {
        std::cerr << "error: CUDA API call : "
                  << cudaGetErrorString(error_code) << std::endl;
        exit(-1);
    }
}

static inline cublasHandle_t& cublas_handle() {
    static cublasHandle_t cublas_handle;
    static bool is_initialized = false;
    if(!is_initialized) {
        auto status = cublasCreate(&cublas_handle);
        if(status != CUBLAS_STATUS_SUCCESS) {
            std::cerr << "error: unable to initialize cublas" << std::endl;
            exit(-1);
        }

        is_initialized = true;
    }

    return cublas_handle;
}

static const char *_cudaGetErrorEnum(cublasStatus_t error)
{
    switch (error)
    {
        case CUBLAS_STATUS_SUCCESS:
            return "CUBLAS_STATUS_SUCCESS";

        case CUBLAS_STATUS_NOT_INITIALIZED:
            return "CUBLAS_STATUS_NOT_INITIALIZED";

        case CUBLAS_STATUS_ALLOC_FAILED:
            return "CUBLAS_STATUS_ALLOC_FAILED";

        case CUBLAS_STATUS_INVALID_VALUE:
            return "CUBLAS_STATUS_INVALID_VALUE";

        case CUBLAS_STATUS_ARCH_MISMATCH:
            return "CUBLAS_STATUS_ARCH_MISMATCH";

        case CUBLAS_STATUS_MAPPING_ERROR:
            return "CUBLAS_STATUS_MAPPING_ERROR";

        case CUBLAS_STATUS_EXECUTION_FAILED:
            return "CUBLAS_STATUS_EXECUTION_FAILED";

        case CUBLAS_STATUS_INTERNAL_ERROR:
            return "CUBLAS_STATUS_INTERNAL_ERROR";
    }

    return "<unknown>";
}
