"""
Train a Neural Network in CUDA from Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - report_cuda_error
#include <cstdio>
#include <cuda_runtime.h>

int report_cuda_error(cudaError_t err, const char* what) {
    // If the CUDA operation succeeded, no error needs to be reported.
    if (err == cudaSuccess) {
        return 0;
    }

    // Print the error message to stderr in the required format.
    fprintf(stderr, "CUDA error in %s: %s\n",
            what, cudaGetErrorString(err));

    // Report the failure to the caller without terminating the program.
    return 1;
}

# Step 2 - device_upload
float* device_upload(const float* host, int n) {
    float* dev = nullptr;

    // Allocate device memory for n floats.
    if (report_cuda_error(
            cudaMalloc(&dev, static_cast<size_t>(n) * sizeof(float)),
            "cudaMalloc (float upload)")) {
        return nullptr;
    }

    // Copy the float array from host memory to device memory.
    if (report_cuda_error(
            cudaMemcpy(dev, host, static_cast<size_t>(n) * sizeof(float),
                       cudaMemcpyHostToDevice),
            "cudaMemcpy (float host to device)")) {
        // Release the allocated memory if the copy fails.
        report_cuda_error(cudaFree(dev), "cudaFree (float upload cleanup)");
        return nullptr;
    }

    // Return the device pointer; the caller owns the buffer.
    return dev;
}

int* device_upload(const int* host, int n) {
    int* dev = nullptr;

    // Allocate device memory for n integers.
    if (report_cuda_error(
            cudaMalloc(&dev, static_cast<size_t>(n) * sizeof(int)),
            "cudaMalloc (int upload)")) {
        return nullptr;
    }

    // Copy the integer array from host memory to device memory.
    if (report_cuda_error(
            cudaMemcpy(dev, host, static_cast<size_t>(n) * sizeof(int),
                       cudaMemcpyHostToDevice),
            "cudaMemcpy (int host to device)")) {
        // Release the allocated memory if the copy fails.
        report_cuda_error(cudaFree(dev), "cudaFree (int upload cleanup)");
        return nullptr;
    }

    // Return the device pointer; the caller owns the buffer.
    return dev;
}

void device_download(float* host, const float* dev, int n) {
    // Copy n floats from device memory back to host memory.
    report_cuda_error(
        cudaMemcpy(host, dev, static_cast<size_t>(n) * sizeof(float),
                   cudaMemcpyDeviceToHost),
        "cudaMemcpy (float device to host)");
}

# Step 3 - make_spiral
#include <cmath>

float lcg_uniform(unsigned* state) {
    // Update the state using a 32-bit linear congruential generator.
    // Unsigned integer arithmetic wraps around on overflow.
    *state = *state * 1664525u + 1013904223u;

    // Use the upper 24 bits to generate a float in [0, 1).
    return (*state >> 8) / 16777216.0f;
}

float lcg_normal(unsigned* state) {
    // Draw two uniform random numbers in the required order.
    float u1 = lcg_uniform(state);
    float u2 = lcg_uniform(state);

    // Box-Muller transform: generate one standard normal sample.
    // The small epsilon prevents taking log(0).
    return sqrtf(-2.0f * logf(u1 + 1e-7f))
           * cosf(6.2831853f * u2);
}

void make_spiral(float* X, int* y, int points_per_class,
                 int classes, float noise, unsigned seed) {
    // Initialize one deterministic random-number generator for the dataset.
    unsigned state = seed;

    // Generate points class by class.
    for (int c = 0; c < classes; ++c) {
        for (int i = 0; i < points_per_class; ++i) {
            // Row-major index of the current point.
            int k = c * points_per_class + i;

            // Radius increases from 0 to 1 within each class.
            float r = (float)i / (float)(points_per_class - 1);

            // Add deterministic Gaussian noise to the spiral angle.
            float t = 4.0f * c + 4.0f * r
                      + noise * lcg_normal(&state);

            // Store the two coordinates in row-major order.
            X[2 * k]     = r * sinf(t);
            X[2 * k + 1] = r * cosf(t);

            // Store the class label for this point.
            y[k] = c;
        }
    }
}

# Step 4 - matmul_kernel
__global__ void matmul_kernel(
    const float* A,
    const float* B,
    float* C,
    int M,
    int N,
    int K
) {
    // Map the x dimension to the output column.
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Map the y dimension to the output row.
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    // Ignore threads outside the M x N output matrix.
    if (row >= M || col >= N) {
        return;
    }

    // Compute one element of C = A * B.
    // A has shape M x K, B has shape K x N, and C has shape M x N.
    float sum = 0.0f;

    for (int k = 0; k < K; ++k) {
        sum += A[row * K + k] * B[k * N + col];
    }

    // Store the result in row-major order.
    C[row * N + col] = sum;
}

void launch_matmul(
    const float* A,
    const float* B,
    float* C,
    int M,
    int N,
    int K
) {
    // Each block contains 16 x 16 = 256 threads.
    dim3 block(16, 16);

    // Round up so the grid covers all rows and columns.
    dim3 grid(
        (N + block.x - 1) / block.x,
        (M + block.y - 1) / block.y
    );

    // Launch the matrix multiplication kernel.
    // This launch is asynchronous; do not synchronize here.
    matmul_kernel<<<grid, block>>>(A, B, C, M, N, K);

    // Check for errors reported by the kernel launch.
    report_cuda_error(cudaGetLastError(), "matmul_kernel launch");
}

# Step 5 - add_bias_kernel
__global__ void add_bias_kernel(float* Z, const float* b, int rows, int cols) {
    // Each thread handles one matrix element.
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Total number of elements in the matrix.
    int total = rows * cols;

    // Ignore threads outside the matrix.
    if (i >= total) {
        return;
    }

    // Broadcast the corresponding column's bias across every row.
    Z[i] += b[i % cols];
}

void launch_add_bias(float* Z, const float* b, int rows, int cols) {
    // Process the entire matrix using 256 threads per block.
    int total = rows * cols;
    int threads_per_block = 256;
    int blocks = (total + threads_per_block - 1) / threads_per_block;

    // Launch the bias-addition kernel.
    add_bias_kernel<<<blocks, threads_per_block>>>(Z, b, rows, cols);

    // Report any immediate kernel-launch error without synchronizing.
    report_cuda_error(cudaGetLastError(), "add_bias_kernel launch");
}

# Step 6 - relu_kernel
__global__ void relu_kernel(float* Z, int n) {
    // Each thread handles one element.
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Ignore threads outside the array.
    if (i >= n) {
        return;
    }

    // ReLU: replace negative values with zero.
    Z[i] = (Z[i] > 0.0f) ? Z[i] : 0.0f;
}

void launch_relu(float* Z, int n) {
    // Use 256 threads per block.
    const int threads_per_block = 256;

    // Calculate enough blocks to cover all n elements.
    const int blocks = (n + threads_per_block - 1) / threads_per_block;

    // Launch the ReLU kernel.
    relu_kernel<<<blocks, threads_per_block>>>(Z, n);

    // Check for immediate kernel-launch errors without synchronizing.
    report_cuda_error(cudaGetLastError(), "relu_kernel launch");
}

# Step 7 - softmax_rows_kernel
__global__ void softmax_rows_kernel(
    const float* logits,
    float* probs,
    int rows,
    int cols
) {
    // Assign one CUDA thread to each row.
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    // Ignore threads outside the matrix.
    if (row >= rows) {
        return;
    }

    int offset = row * cols;

    // Find the maximum logit in this row for numerical stability.
    float max_val = logits[offset];

    for (int j = 1; j < cols; ++j) {
        float value = logits[offset + j];
        if (value > max_val) {
            max_val = value;
        }
    }

    // Compute the sum of exponentials after subtracting the maximum.
    float sum_exp = 0.0f;

    for (int j = 0; j < cols; ++j) {
        sum_exp += expf(logits[offset + j] - max_val);
    }

    // Normalize the exponentials to obtain probabilities.
    // This also works in place when probs == logits because the sum
    // has already been computed before any output element is overwritten.
    for (int j = 0; j < cols; ++j) {
        probs[offset + j] =
            expf(logits[offset + j] - max_val) / sum_exp;
    }
}

void launch_softmax_rows(
    const float* logits,
    float* probs,
    int rows,
    int cols
) {
    // Use 128 threads per block, with one thread per row.
    const int threads_per_block = 128;

    // Calculate enough blocks to cover all rows.
    const int blocks =
        (rows + threads_per_block - 1) / threads_per_block;

    // Launch the softmax kernel without synchronizing.
    softmax_rows_kernel<<<blocks, threads_per_block>>>(
        logits, probs, rows, cols
    );

    // Report any immediate kernel-launch error.
    report_cuda_error(
        cudaGetLastError(),
        "softmax_rows_kernel launch"
    );
}

# Step 8 - cross_entropy_kernel
__global__ void cross_entropy_kernel(
    const float* probs,
    const int* labels,
    int rows,
    int cols,
    float* total
) {
    // Shared memory for the 256 loss values in this block.
    __shared__ float shared_sum[256];

    int tid = threadIdx.x;
    int r = blockIdx.x * blockDim.x + tid;

    // Threads without a corresponding sample contribute zero.
    float loss = 0.0f;

    if (r < rows) {
        // Obtain the probability assigned to the correct class.
        int label = labels[r];
        float p = probs[r * cols + label];

        // Avoid log(0) and compute the sample's cross-entropy loss.
        loss = -logf(fmaxf(p, 1e-12f));
    }

    // Store this thread's loss in shared memory.
    shared_sum[tid] = loss;
    __syncthreads();

    // Reduce the 256 values using a binary tree.
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
        }

        // Ensure all updates at this level finish before the next level.
        __syncthreads();
    }

    // Add this block's sum to the global total.
    if (tid == 0) {
        atomicAdd(total, shared_sum[0]);
    }
}

float cross_entropy_loss(
    const float* probs,
    const int* labels,
    int rows,
    int cols
) {
    // Avoid launching an invalid grid or dividing by zero.
    if (rows <= 0 || cols <= 0) {
        return 0.0f;
    }

    // Allocate a device scalar to accumulate the total loss.
    float* d_total = nullptr;

    if (report_cuda_error(
            cudaMalloc(reinterpret_cast<void**>(&d_total), sizeof(float)),
            "cudaMalloc cross-entropy total")) {
        return 0.0f;
    }

    // Initialize the global accumulator to zero.
    if (report_cuda_error(
            cudaMemset(d_total, 0, sizeof(float)),
            "cudaMemset cross-entropy total")) {
        report_cuda_error(
            cudaFree(d_total),
            "cudaFree cross-entropy total after memset failure");
        return 0.0f;
    }

    // Use 256 threads per block, with one thread per sample.
    const int threads_per_block = 256;
    const int blocks =
        (rows + threads_per_block - 1) / threads_per_block;

    // Launch the cross-entropy reduction kernel.
    cross_entropy_kernel<<<blocks, threads_per_block>>>(
        probs, labels, rows, cols, d_total
    );

    // Check for an immediate kernel-launch error.
    if (report_cuda_error(
            cudaGetLastError(),
            "cross_entropy_kernel launch")) {
        report_cuda_error(
            cudaFree(d_total),
            "cudaFree cross-entropy total after launch failure");
        return 0.0f;
    }

    // Copying the result back also synchronizes the preceding GPU work.
    float total = 0.0f;

    if (report_cuda_error(
            cudaMemcpy(
                &total,
                d_total,
                sizeof(float),
                cudaMemcpyDeviceToHost
            ),
            "cudaMemcpy cross-entropy total to host")) {
        report_cuda_error(
            cudaFree(d_total),
            "cudaFree cross-entropy total after copy failure");
        return 0.0f;
    }

    // Release the temporary device allocation.
    report_cuda_error(
        cudaFree(d_total),
        "cudaFree cross-entropy total"
    );

    // Return the mean loss across all samples.
    return total / static_cast<float>(rows);
}

# Step 9 - init_mlp (not yet solved)
# TODO: implement

# Step 10 - alloc_workspace (not yet solved)
# TODO: implement

# Step 11 - forward_mlp (not yet solved)
# TODO: implement

# Step 12 - softmax_ce_grad_kernel (not yet solved)
# TODO: implement

# Step 13 - matmul_tn_kernel (not yet solved)
# TODO: implement

# Step 14 - matmul_nt_kernel (not yet solved)
# TODO: implement

# Step 15 - column_sum_kernel (not yet solved)
# TODO: implement

# Step 16 - relu_backward_kernel (not yet solved)
# TODO: implement

# Step 17 - backward_mlp (not yet solved)
# TODO: implement

# Step 18 - sgd_update_kernel (not yet solved)
# TODO: implement

# Step 19 - train_step (not yet solved)
# TODO: implement

# Step 20 - train_mlp (not yet solved)
# TODO: implement

# Step 21 - accuracy_kernel (not yet solved)
# TODO: implement

# Step 22 - time_train_step_ms (not yet solved)
# TODO: implement

