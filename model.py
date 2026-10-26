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

# Step 4 - matmul_kernel (not yet solved)
# TODO: implement

# Step 5 - add_bias_kernel (not yet solved)
# TODO: implement

# Step 6 - relu_kernel (not yet solved)
# TODO: implement

# Step 7 - softmax_rows_kernel (not yet solved)
# TODO: implement

# Step 8 - cross_entropy_kernel (not yet solved)
# TODO: implement

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

