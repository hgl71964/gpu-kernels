#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <stdio.h>

#define MATRIX_M 1024
#define MATRIX_N 1024

__global__ void transpose_naive(float* input, float* output) {
    int col_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int row_idx = blockIdx.y * blockDim.y + threadIdx.y;

    if (col_idx < MATRIX_N && row_idx < MATRIX_M) {

        // origin
        int idx = row_idx * MATRIX_N + col_idx;

        // target
        int trans_idx = col_idx * MATRIX_M + row_idx;

        // read is consecutive; write is non-consecutive
        output[trans_idx] = input[idx];
    }
}

__global__ void vector_transpose(float* input, float* output)
{
    int col_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int row_idx = blockIdx.y * blockDim.y + threadIdx.y;

    // 这里每个线程处理4行，4列，因此需要除以4
    if (col_idx >= (MATRIX_N >> 2) || row_idx >= (MATRIX_M >> 2)) {
        return;
    }

    // 找到正确的地址
    int offset = (row_idx * MATRIX_N + col_idx) << 2; // * 4
    float4* input_v4 = reinterpret_cast<float4*>(input + offset);

    // 以向量的方式拿到4 * 4矩阵
    float4 src_row0 = input_v4[0];
    float4 src_row1 = input_v4[MATRIX_N >> 2];
    float4 src_row2 = input_v4[MATRIX_N >> 1];
    float4 src_row3 = input_v4[(MATRIX_N >> 2) * 3];


    // 4 * 4小矩阵转置
    float4 dst_row0 = make_float4(src_row0.x, src_row1.x, src_row2.x, src_row3.x);
    float4 dst_row1 = make_float4(src_row0.y, src_row1.y, src_row2.y, src_row3.y);
    float4 dst_row2 = make_float4(src_row0.z, src_row1.z, src_row2.z, src_row3.z);
    float4 dst_row3 = make_float4(src_row0.w, src_row1.w, src_row2.w, src_row3.w);

    // 将4行写入
    offset = (col_idx * MATRIX_M + row_idx) << 2;
    float4* dst_v4 = reinterpret_cast<float4*>(output + offset);
    dst_v4[0] = dst_row0;
    dst_v4[MATRIX_M >> 2] = dst_row1;
    dst_v4[MATRIX_M >> 1] = dst_row2;
    dst_v4[(MATRIX_M >> 2) * 3] = dst_row3;
}

int main() {
    float* input = (float*)malloc(sizeof(float) * MATRIX_M * MATRIX_N);
    float* output = (float*)malloc(sizeof(float) * MATRIX_M * MATRIX_N);
    return 0;
}