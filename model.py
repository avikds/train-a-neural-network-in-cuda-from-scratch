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

# Step 9 - init_mlp
#include <vector>  // Newly required for host-side parameter arrays

struct MLP {
    int in_dim, hidden, out_dim;
    float *W1, *b1, *W2, *b2;  // Device pointers
};

// Forward declaration so init_mlp can clean up partial allocations.
void free_mlp(MLP& m);

MLP init_mlp(int in_dim, int hidden, int out_dim, unsigned seed) {
    // Store dimensions and initialize all device pointers to nullptr.
    MLP m = {
        in_dim, hidden, out_dim,
        nullptr, nullptr, nullptr, nullptr
    };

    // Host-side arrays, stored in row-major order.
    std::vector<float> hW1(in_dim * hidden);
    std::vector<float> hb1(hidden, 0.0f);
    std::vector<float> hW2(hidden * out_dim);
    std::vector<float> hb2(out_dim, 0.0f);

    // Use one deterministic generator for both weight matrices.
    unsigned state = seed;

    // Initialize W1 using the specified fan-in scaling.
    const float scale1 = sqrtf(6.0f / static_cast<float>(in_dim));

    for (int i = 0; i < in_dim * hidden; ++i) {
        hW1[i] = (2.0f * lcg_uniform(&state) - 1.0f) * scale1;
    }

    // Initialize W2 next, continuing with the same RNG state.
    const float scale2 = sqrtf(6.0f / static_cast<float>(hidden));

    for (int i = 0; i < hidden * out_dim; ++i) {
        hW2[i] = (2.0f * lcg_uniform(&state) - 1.0f) * scale2;
    }

    // Upload all four arrays to device memory.
    m.W1 = device_upload(hW1.data(), in_dim * hidden);
    m.b1 = device_upload(hb1.data(), hidden);
    m.W2 = device_upload(hW2.data(), hidden * out_dim);
    m.b2 = device_upload(hb2.data(), out_dim);

    // If any upload fails, release all successful allocations.
    if (m.W1 == nullptr || m.b1 == nullptr ||
        m.W2 == nullptr || m.b2 == nullptr) {
        free_mlp(m);
    }

    return m;
}

void free_mlp(MLP& m) {
    // Release each device allocation and clear its pointer.
    if (m.W1 != nullptr) {
        report_cuda_error(cudaFree(m.W1), "cudaFree W1");
        m.W1 = nullptr;
    }

    if (m.b1 != nullptr) {
        report_cuda_error(cudaFree(m.b1), "cudaFree b1");
        m.b1 = nullptr;
    }

    if (m.W2 != nullptr) {
        report_cuda_error(cudaFree(m.W2), "cudaFree W2");
        m.W2 = nullptr;
    }

    if (m.b2 != nullptr) {
        report_cuda_error(cudaFree(m.b2), "cudaFree b2");
        m.b2 = nullptr;
    }
}

# Step 10 - alloc_workspace
struct Workspace {
    int batch;
    float *H, *P;                    // Forward: hidden activations, probabilities
    float *dZ2, *dH;                 // Backward: logits and hidden gradients
    float *dW1, *db1, *dW2, *db2;    // Parameter gradients
};

// Forward declaration for cleanup on allocation failure.
void free_workspace(Workspace& w);

Workspace alloc_workspace(const MLP& m, int batch) {
    Workspace w = {
        batch,
        nullptr, nullptr,
        nullptr, nullptr,
        nullptr, nullptr, nullptr, nullptr
    };

    // Allocate a float buffer on the GPU and initialize it to zero.
    // Returns false if either allocation or initialization fails.
    auto alloc_zero = [](float*& ptr,
                         size_t count,
                         const char* alloc_name,
                         const char* memset_name) -> bool {
        size_t bytes = count * sizeof(float);

        if (report_cuda_error(
                cudaMalloc(reinterpret_cast<void**>(&ptr), bytes),
                alloc_name)) {
            return false;
        }

        if (report_cuda_error(
                cudaMemset(ptr, 0, bytes),
                memset_name)) {
            return false;
        }

        return true;
    };

    // Forward-pass buffers.
    if (!alloc_zero(
            w.H,
            static_cast<size_t>(batch) * m.hidden,
            "cudaMalloc workspace H",
            "cudaMemset workspace H")) {
        free_workspace(w);
        return w;
    }

    if (!alloc_zero(
            w.P,
            static_cast<size_t>(batch) * m.out_dim,
            "cudaMalloc workspace P",
            "cudaMemset workspace P")) {
        free_workspace(w);
        return w;
    }

    // Backward-pass buffers.
    if (!alloc_zero(
            w.dZ2,
            static_cast<size_t>(batch) * m.out_dim,
            "cudaMalloc workspace dZ2",
            "cudaMemset workspace dZ2")) {
        free_workspace(w);
        return w;
    }

    if (!alloc_zero(
            w.dH,
            static_cast<size_t>(batch) * m.hidden,
            "cudaMalloc workspace dH",
            "cudaMemset workspace dH")) {
        free_workspace(w);
        return w;
    }

    // Parameter-gradient buffers.
    if (!alloc_zero(
            w.dW1,
            static_cast<size_t>(m.in_dim) * m.hidden,
            "cudaMalloc workspace dW1",
            "cudaMemset workspace dW1")) {
        free_workspace(w);
        return w;
    }

    if (!alloc_zero(
            w.db1,
            static_cast<size_t>(m.hidden),
            "cudaMalloc workspace db1",
            "cudaMemset workspace db1")) {
        free_workspace(w);
        return w;
    }

    if (!alloc_zero(
            w.dW2,
            static_cast<size_t>(m.hidden) * m.out_dim,
            "cudaMalloc workspace dW2",
            "cudaMemset workspace dW2")) {
        free_workspace(w);
        return w;
    }

    if (!alloc_zero(
            w.db2,
            static_cast<size_t>(m.out_dim),
            "cudaMalloc workspace db2",
            "cudaMemset workspace db2")) {
        free_workspace(w);
        return w;
    }

    return w;
}

void free_workspace(Workspace& w) {
    // Release each allocated buffer and clear its pointer.
    if (w.H != nullptr) {
        report_cuda_error(cudaFree(w.H), "cudaFree workspace H");
        w.H = nullptr;
    }

    if (w.P != nullptr) {
        report_cuda_error(cudaFree(w.P), "cudaFree workspace P");
        w.P = nullptr;
    }

    if (w.dZ2 != nullptr) {
        report_cuda_error(cudaFree(w.dZ2), "cudaFree workspace dZ2");
        w.dZ2 = nullptr;
    }

    if (w.dH != nullptr) {
        report_cuda_error(cudaFree(w.dH), "cudaFree workspace dH");
        w.dH = nullptr;
    }

    if (w.dW1 != nullptr) {
        report_cuda_error(cudaFree(w.dW1), "cudaFree workspace dW1");
        w.dW1 = nullptr;
    }

    if (w.db1 != nullptr) {
        report_cuda_error(cudaFree(w.db1), "cudaFree workspace db1");
        w.db1 = nullptr;
    }

    if (w.dW2 != nullptr) {
        report_cuda_error(cudaFree(w.dW2), "cudaFree workspace dW2");
        w.dW2 = nullptr;
    }

    if (w.db2 != nullptr) {
        report_cuda_error(cudaFree(w.db2), "cudaFree workspace db2");
        w.db2 = nullptr;
    }
}

# Step 11 - forward_mlp
void forward_mlp(const MLP& m, const float* X, int batch, Workspace& w) {
    // Layer 1: H = X @ W1
    // X:  batch x in_dim
    // W1: in_dim x hidden
    // H:  batch x hidden
    launch_matmul(X, m.W1, w.H, batch, m.hidden, m.in_dim);

    // Add the first layer's bias to every row.
    launch_add_bias(w.H, m.b1, batch, m.hidden);

    // Apply ReLU to the hidden activations in place.
    launch_relu(w.H, batch * m.hidden);

    // Layer 2: P = H @ W2
    // H:  batch x hidden
    // W2: hidden x out_dim
    // P:  batch x out_dim
    launch_matmul(w.H, m.W2, w.P, batch, m.out_dim, m.hidden);

    // Add the output layer's bias to every row.
    launch_add_bias(w.P, m.b2, batch, m.out_dim);

    // Convert output logits into class probabilities in place.
    launch_softmax_rows(w.P, w.P, batch, m.out_dim);
}

# Step 12 - softmax_ce_grad_kernel
__global__ void softmax_ce_grad_kernel(
    const float* P,
    const int* labels,
    float* dZ,
    int rows,
    int cols
) {
    // Each thread processes one element of the rows x cols matrix.
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int total = rows * cols;

    // Ignore threads outside the matrix.
    if (idx >= total) {
        return;
    }

    // Convert the linear index to a row and column.
    int r = idx / cols;
    int j = idx % cols;

    // For the correct class, subtract 1 from its predicted probability.
    // For all other classes, retain the predicted probability.
    float target = (j == labels[r]) ? 1.0f : 0.0f;

    // Gradient of the mean softmax cross-entropy loss.
    dZ[idx] = (P[idx] - target) / static_cast<float>(rows);
}

void launch_softmax_ce_grad(
    const float* P,
    const int* labels,
    float* dZ,
    int rows,
    int cols
) {
    // Use 256 threads per block.
    const int threads_per_block = 256;

    // Calculate the number of blocks needed for all matrix elements.
    int total = rows * cols;
    int blocks = (total + threads_per_block - 1) / threads_per_block;

    // Launch the gradient kernel.
    softmax_ce_grad_kernel<<<blocks, threads_per_block>>>(
        P, labels, dZ, rows, cols
    );

    // Report any immediate kernel-launch error without synchronizing.
    report_cuda_error(
        cudaGetLastError(),
        "softmax_ce_grad_kernel launch"
    );
}

# Step 13 - matmul_tn_kernel
__global__ void matmul_tn_kernel(
    const float* A,
    const float* D,
    float* C,
    int R,
    int I,
    int J
) {
    // Map CUDA thread coordinates to the output matrix C.
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    // C has dimensions I x J.
    if (i >= I || j >= J) {
        return;
    }

    // Compute C[i][j] = sum_r A[r][i] * D[r][j].
    float sum = 0.0f;

    for (int r = 0; r < R; ++r) {
        // A is stored as R x I, and D is stored as R x J.
        sum += A[r * I + i] * D[r * J + j];
    }

    // Store the output in row-major order.
    C[i * J + j] = sum;
}

void launch_matmul_tn(
    const float* A,
    const float* D,
    float* C,
    int R,
    int I,
    int J
) {
    // Use 16 x 16 threads per block.
    dim3 block(16, 16);

    // Cover all J output columns and I output rows.
    dim3 grid(
        (J + block.x - 1) / block.x,
        (I + block.y - 1) / block.y
    );

    // Launch the transposed matrix multiplication kernel.
    // No explicit synchronization is performed here.
    matmul_tn_kernel<<<grid, block>>>(A, D, C, R, I, J);

    // Report immediate kernel-launch errors.
    report_cuda_error(
        cudaGetLastError(),
        "matmul_tn_kernel launch"
    );
}

# Step 14 - matmul_nt_kernel
__global__ void matmul_nt_kernel(
    const float* D,
    const float* W,
    float* C,
    int R,
    int J,
    int I
) {
    // Map the x dimension to the output column i.
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Map the y dimension to the output row r.
    int r = blockIdx.y * blockDim.y + threadIdx.y;

    // C has dimensions R x I.
    if (r >= R || i >= I) {
        return;
    }

    // Compute C[r][i] = sum_j D[r][j] * W[i][j].
    float sum = 0.0f;

    for (int j = 0; j < J; ++j) {
        sum += D[r * J + j] * W[i * J + j];
    }

    // Store the result in row-major order.
    C[r * I + i] = sum;
}

void launch_matmul_nt(
    const float* D,
    const float* W,
    float* C,
    int R,
    int J,
    int I
) {
    // Use 16 x 16 threads per block.
    dim3 block(16, 16);

    // Cover all I output columns and R output rows.
    dim3 grid(
        (I + block.x - 1) / block.x,
        (R + block.y - 1) / block.y
    );

    // Launch the kernel without explicit synchronization.
    matmul_nt_kernel<<<grid, block>>>(D, W, C, R, J, I);

    // Report immediate kernel-launch errors.
    report_cuda_error(
        cudaGetLastError(),
        "matmul_nt_kernel launch"
    );
}

# Step 15 - column_sum_kernel
__global__ void column_sum_kernel(
    const float* D,
    float* out,
    int rows,
    int cols
) {
    // Assign one CUDA thread to each column.
    int c = blockIdx.x * blockDim.x + threadIdx.x;

    // Ignore threads outside the matrix's column range.
    if (c >= cols) {
        return;
    }

    // Sum all elements in column c.
    float sum = 0.0f;

    for (int r = 0; r < rows; ++r) {
        sum += D[r * cols + c];
    }

    // Overwrite the output with this column's sum.
    out[c] = sum;
}

void launch_column_sum(
    const float* D,
    float* out,
    int rows,
    int cols
) {
    // Use 128 threads per block.
    const int threads_per_block = 128;

    // Calculate enough blocks to cover every column.
    const int blocks =
        (cols + threads_per_block - 1) / threads_per_block;

    // Launch the column-summation kernel.
    column_sum_kernel<<<blocks, threads_per_block>>>(
        D, out, rows, cols
    );

    // Report immediate kernel-launch errors without synchronizing.
    report_cuda_error(
        cudaGetLastError(),
        "column_sum_kernel launch"
    );
}

# Step 16 - relu_backward_kernel
__global__ void relu_backward_kernel(float* dH, const float* H, int n) {
    // Each CUDA thread processes one element.
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Ignore threads outside the array.
    if (i >= n) {
        return;
    }

    // ReLU derivative:
    // Preserve the gradient where the forward activation was positive.
    // Otherwise, set the gradient to zero.
    if (H[i] <= 0.0f) {
        dH[i] = 0.0f;
    }
}

void launch_relu_backward(float* dH, const float* H, int n) {
    // Avoid launching a zero-sized grid.
    if (n <= 0) {
        return;
    }

    // Use 256 threads per block.
    const int threads_per_block = 256;

    // Calculate enough blocks to cover all n elements.
    const int blocks = (n + threads_per_block - 1) / threads_per_block;

    // Launch the backward ReLU kernel.
    relu_backward_kernel<<<blocks, threads_per_block>>>(dH, H, n);

    // Report immediate kernel-launch errors without synchronizing.
    report_cuda_error(
        cudaGetLastError(),
        "relu_backward_kernel launch"
    );
}

# Step 17 - backward_mlp
void backward_mlp(
    const MLP& m,
    const float* X,
    const int* labels,
    int batch,
    Workspace& w
) {
    // 1. Gradient of mean cross-entropy loss with respect to output logits.
    // dZ2: batch x out_dim
    launch_softmax_ce_grad(
        w.P, labels, w.dZ2, batch, m.out_dim
    );

    // 2. Second-layer weight gradient: dW2 = H^T @ dZ2.
    // H:   batch x hidden
    // dZ2: batch x out_dim
    // dW2: hidden x out_dim
    launch_matmul_tn(
        w.H, w.dZ2, w.dW2,
        batch, m.hidden, m.out_dim
    );

    // Second-layer bias gradient: db2 = column_sum(dZ2).
    launch_column_sum(
        w.dZ2, w.db2, batch, m.out_dim
    );

    // 3. Propagate the gradient through the second-layer weights.
    // dH = dZ2 @ W2^T
    // dH: batch x hidden
    launch_matmul_nt(
        w.dZ2, m.W2, w.dH,
        batch, m.out_dim, m.hidden
    );

    // Apply the derivative of ReLU using saved post-ReLU activations H.
    // Gradients are zeroed wherever H <= 0.
    launch_relu_backward(
        w.dH, w.H, batch * m.hidden
    );

    // 4. First-layer weight gradient: dW1 = X^T @ dH.
    // X:   batch x in_dim
    // dH:  batch x hidden
    // dW1: in_dim x hidden
    launch_matmul_tn(
        X, w.dH, w.dW1,
        batch, m.in_dim, m.hidden
    );

    // First-layer bias gradient: db1 = column_sum(dH).
    launch_column_sum(
        w.dH, w.db1, batch, m.hidden
    );
}

# Step 18 - sgd_update_kernel
__global__ void sgd_update_kernel(
    float* param,
    const float* grad,
    float lr,
    int n
) {
    // Assign one CUDA thread to each parameter.
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Ignore threads outside the parameter array.
    if (i >= n) {
        return;
    }

    // Gradient descent update: param[i] -= learning_rate * gradient[i].
    param[i] -= lr * grad[i];
}

void launch_sgd_update(
    float* param,
    const float* grad,
    float lr,
    int n
) {
    // Use 256 threads per block.
    const int threads_per_block = 256;

    // Avoid launching a zero-sized grid.
    if (n <= 0) {
        return;
    }

    // Calculate enough blocks to cover all n parameters.
    const int blocks = (n + threads_per_block - 1) / threads_per_block;

    // Launch the SGD update kernel.
    sgd_update_kernel<<<blocks, threads_per_block>>>(
        param, grad, lr, n
    );

    // Report immediate kernel-launch errors without synchronizing.
    report_cuda_error(
        cudaGetLastError(),
        "sgd_update_kernel launch"
    );
}

# Step 19 - train_step
float train_step(
    MLP& m,
    Workspace& w,
    const float* X,
    const int* labels,
    int batch,
    float lr
) {
    // 1. Forward pass: compute hidden activations and class probabilities.
    forward_mlp(m, X, batch, w);

    // 2. Compute the loss before updating the parameters.
    float loss = cross_entropy_loss(w.P, labels, batch, m.out_dim);

    // 3. Backward pass: compute gradients for all four parameters.
    backward_mlp(m, X, labels, batch, w);

    // 4. Update W1 using its gradient.
    launch_sgd_update(
        m.W1, w.dW1, lr, m.in_dim * m.hidden
    );

    // Update the first-layer bias.
    launch_sgd_update(
        m.b1, w.db1, lr, m.hidden
    );

    // Update W2 using its gradient.
    launch_sgd_update(
        m.W2, w.dW2, lr, m.hidden * m.out_dim
    );

    // Update the second-layer bias.
    launch_sgd_update(
        m.b2, w.db2, lr, m.out_dim
    );

    // Return the loss measured before the parameter updates.
    return loss;
}

# Step 20 - train_mlp
float train_mlp(
    MLP& m,
    Workspace& w,
    const float* X,
    const int* labels,
    int batch,
    float lr,
    int epochs,
    int print_every
) {
    // No training occurs when epochs <= 0.
    // Return zero and leave the model unchanged.
    if (epochs <= 0) {
        return 0.0f;
    }

    float last_loss = 0.0f;

    // Run the requested number of full-batch training steps.
    // Epoch numbering starts at 1.
    for (int epoch = 1; epoch <= epochs; ++epoch) {
        // Execute one complete training step and retain its loss.
        last_loss = train_step(m, w, X, labels, batch, lr);

        // Print at epoch 1 and every print_every epochs,
        // but only when print_every is positive.
        if (print_every > 0 &&
            (epoch == 1 || epoch % print_every == 0)) {
            std::printf(
                "epoch %5d  loss %.4f\n",
                epoch,
                last_loss
            );
        }
    }

    // Return the loss from the final training step.
    return last_loss;
}

# Step 21 - accuracy_kernel (not yet solved)
# TODO: implement

# Step 22 - time_train_step_ms (not yet solved)
# TODO: implement

