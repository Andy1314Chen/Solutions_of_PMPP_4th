## Chapter 3: Multidimensional grids and data 

## Exercises

1. In this chapter we implemented a matrix multiplication kernel that has each thread produce one output matrix element. In this question, you will implement different matrix-matrix multiplication kernels and compare them.

- a. Write a kernel that has each thread produce one output matrix row. Fill in the execution configuration parameters for the design.

```c
__global__ foo_kernel(float* d_A, float* d_B, float* d_C, size_t n) {
    size_t row = blockDim.x * blockIdx.x + threadIdx.x;
    if (row < n) {
        for (size_t col = 0; col < n; col++) {
            float val = 0.0f;
            for (size_t k = 0; k < n; k++) {
                val += d_A[row * n + k] * d_B[k * n + col];
            }
            d_C[row * n + col] = val;
        }
    }
}
```

- b. Write a kernel that has each thread produce one output matrix column. Fill in the execution configuration parameters for the design.

```c
__global__ foo_kernel(float* d_A, float* d_B, float* d_C, size_t n) {
    size_t col = blockDim.x * blockIdx.x + threadIdx.x;
    if (col < n) {
        for (size_t row = 0; row < n; row++) {
            float val = 0.0f;
            for (size_t k = 0; k < n; k++) {
                val += d_A[row * n + k] * d_B[k * n + col];
            }
            d_C[row * n + col] = val;
        }
    }
}
```

- c. Analyze the pros and cons of each of the two kernel designs.

```c
Kernel 1: Each Thread Produces One Output Matrix Row
Pros:
- Simplicity: Easy to implement and understand.
- Memory Access: Coalesced memory access for each row, improving performance.

Cons:
- Imbalanced Workloads: Threads may have uneven workloads if rows vary in computation.
- Scalability: Limited by the number of rows, potentially leading to register pressure.

Kernel 2: Each Thread Produces One Output Matrix Column
Pros:
- Simplicity: Easy to implement and understand.
- Memory Access: Coalesced writes to matrix C.

Cons:
- Memory Access: Non-coalesced memory access for reading matrix B, reducing performance.
- Imbalanced Workloads: Similar issues with threads having uneven workloads.

Summary
- Row-wise Kernel: Better memory access patterns but may have scalability issues.
- Column-wise Kernel: Easier to implement but suffers from inefficient memory access.
```

___
2. A matrix-vector multiplication takes an input matrix B and a vector C and produces one output vector A. Each element of the output vector A is the dot product of one row of the input matrix B and C, that is, A[i] = sum_{j=0}(B[i][j] * C[j]). For simplicity we will handle only square matrices whose elements are singleprecision floating-point numbers. Write a matrix-vector multiplication kernel and the host stub function that can be called with four parameters: pointer to the output matrix, pointer to the input matrix, pointer to the input vector, and the number of elements in each dimension. Use one thread to calculate an output vector element.

```c
__global__ void foo_kernel(float* d_A, float* d_B, float* d_C, size_t n) {
    size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < n) {
        float val = 0.0f;
        for (size_t i = 0; i < n; i++) {
            val += d_B[idx * n + i] * d_C[i];
        }
        d_A[idx] = val;
    }
}

void foo(float* d_A, float* d_B, float* d_C, size_t n) {
    dim3 blockDim(256);
    dim3 gridDim((n + 256 - 1)/256);

    foo_kernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, n);
}
```


___
3. Consider the following CUDA kernel and the corresponding host function that calls it:

```c
 __global__ void foo_kernel(float* a, float* b, unsigned int M, unsigned int N) {
    unsigned int row = blockIdx.y*blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x*blockDim.x + threadIdx.x;
    if (row < M && col < N){
        b[row*N + col] = a[row*N + col]/2.1f + 4.8f;
    }
}
void foo(float* a_d, float* b_d) {
    unsigned int M = 150;
    unsigned int N = 300;
    dim3 bd(16, 32);
    dim3 gd((N - 1)/16 + 1, (M - 1)/32 + 1);
    foo_kernel<<<gd, bd>>>(a_d, b_d, M, N);
}
```
- a. What is the number of threads per block?
```c
number of threads per block: 16 * 32 = 512
```
- b. What is the number of threads in the grid?
```
512 * number of blocks = 512 * ((N - 1)/16 + 1) * ((M - 1)/32 + 1) = 48640
```
- c. What is the number of blocks in the grid?
```c
number of blocks: ((N - 1)/16 + 1) * ((M - 1)/32 + 1) = 95
```
- d. What is the number of threads that execute the code on line 05?
```c
number of threads: 150 * 300 = 45000
```

___
4. Consider a 2D matrix with a width of 400 and a height of 500. The matrix is stored as a one-dimensional array. Specify the array index of the matrix element at row 20 and column 10:
- a. If the matrix is stored in row-major order.
```c
idx = y * width + x = 20 * 400 + 10 = 8010
```
- b. If the matrix is stored in column-major order.
```c
idx = x * height + y = 10 * 500 + 20 = 5020
```

___
5. Consider a 3D tensor with a width of 400, a height of 500, and a depth of 300. The tensor is stored as a one-dimensional array in row-major order. Specify the array index of the tensor element at x=10, y=20, and z=5.

```c
idx = z * width * height + x * width + y = 5 * (400 * 500) + 10 * 400 + 20 = 1004020
```
