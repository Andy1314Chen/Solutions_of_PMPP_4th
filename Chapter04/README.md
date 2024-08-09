## Chapter 4: Compute architecture and scheduling

## Exercises

1. Consider the following CUDA kernel and the corresponding host function that calls it:

```c
__global__ void foo_kernel(int* a, int* b) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (threadIdx.x < 40 || threadIdx.x >= 104) {
        b[i] = a[i] + 1;                                // 04
    }
    if (i%2 == 0) {
        a[i] = b[i] * 2;                                // 07
    }
    for (unsigned int j = 0; j < 5 - (i%3); j++) {      // 09
        b[i] += j;
    }
}

void foo(int* a_d, int* b_d) {
    unsigned int N = 1024;
    foo_kernel<<<(N + 128 - 1)/128, 128>>>(a_d, b_d);
}
```

- a. What is the number of warps per block?
```c
128/32 = 4
```
- b. What is the number of warps in the grid?
```c
1024/32 = 32
```
- c. For the statement on line 04:
    - i. How many warps in the grid are active?
    ```c
        threadIdx.x <   40: 2 warps
        threadIdx.x >= 104: 1 warp
                        --> 3 warps * 8 = 24 warps
    ```
    - ii. How many warps in the grid are divergent?
    ```c
        2 warps * 8 = 16 warps
    ```
    - iii. What is the SIMD efficiency (in %) of warp 0 of block 0?
    ```c
        100%
    ```
    - iv. What is the SIMD efficiency (in %) of warp 1 of block 0?
    ```c
        (40 - 32)/32 = 25%
    ```
    - v. What is the SIMD efficiency (in %) of warp 3 of block 0?
    ```c
        (128 - 104)/32 = 75%
    ```

- d. For the statement on line 07:
    - i. How many warps in the grid are active?
    ```c
        32 warps
    ```
    - ii. How many warps in the grid are divergent?
    ```c
        32 warps
    ```
    - iii. What is the SIMD efficiency (in %) of warp 0 of block 0?
    ```c
        50%
    ```

- e. For the loop on line 09:
    - i. How many iterations have no divergence?
    ```c
        341 (for i % 3 == 0), 341 (for i % 3 == 1), 342 (for i % 3 == 2)
    ```
    - ii. How many iterations have divergence?
    ```c
        Any warp that has mixed i % 3 values will cause divergence. Generally, since each wave of 32 threads could be mixed, the number of divergent iterations would be significant.
    ```

2. For a vector addition, assume that the vector length is 2000, each thread calculates one output element, and the thread block size is 512 threads. How many threads will be in the grid?

```python
    ((2000 + 512 - 1)//512) * 512 = 2048
```

3. For the previous question, how many warps do you expect to have divergence due to the boundary check on vector length?

```python
    1 warp
```

4. Consider a hypothetical block with 8 threads executing a section of code before reaching a barrier. The threads require the following amount of time (in microseconds) to execute the sections: 2.0, 2.3, 3.0, 2.8, 2.4, 1.9, 2.6, and 2.9; they spend the rest of their time waiting for the barrier. What percentage of the threads' total execution time is spent waiting for the barrier?

```python
    (3.0 * 8 - (2.0 + 2.3 + 3.0 + 2.8 + 2.4 + 1.9 + 2.6 + 2.9))/(2.0 + 2.3 + 3.0 + 2.8 + 2.4 + 1.9 + 2.6 + 2.9) * 100 = 20.60%
```

5. A CUDA programmer says that if they launch a kernel with only 32 threads in each block, they can leave out the __syncthreads() instruction wherever barrier synchronization is needed. Do you think this is a good idea? Explain.

```python
    While it might seem that omitting __syncthreads() when using 32 threads per block could work due to the lockstep execution of threads within a warp, this is not a good practice. The correct approach is to always use __syncthreads() when barrier synchronization is needed to ensure correct and maintainable code. This practice aligns with the principle of writing robust and future-proof software.
```

6. If a CUDA device's SM can take up to 1536 threads and up to 4 thread blocks, which of the following block configurations would result in the most number of threads in the SM?
a. 128 threads per block
b. 256 threads per block
c. 512 threads per block
d. 1024 threads per block

```python
    c
```

7. Assume a device that allows up to 64 blocks per SM and 2048 threads per SM. Indicate which of the following assignments per SM are possible. In the cases in which it is possible, indicate the occupancy level.
a. 8 blocks with 128 threads each
b. 16 blocks with 64 threads each
c. 32 blocks with 32 threads each
d. 64 blocks with 32 threads each
e. 32 blocks with 64 threads each

```python
    a. 8 * 128 / 2048 = 50%
    b. 16 * 64 / 2048 = 50%
    c. 32 * 32 / 2048 = 50%
    d. 64 * 32 / 2048 = 100%
    e. 32 * 64 / 2048 = 100%
```

8. Consider a GPU with the following hardware limits: 2048 threads per SM, 32 blocks per SM, and 64K (65,536) registers per SM. For each of the following kernel characteristics, specify whether the kernel can achieve full occupancy. If not, specify the limiting factor.
a. The kernel uses 128 threads per block and 30 registers per thread.
b. The kernel uses 32 threads per block and 29 registers per thread.
c. The kernel uses 256 threads per block and 34 registers per thread.

```python
    a. blocks: 16, occupancy: 100%, 16 * 128 * 30 < 65,536, limiting factor: threads
    b. blocks: 32, occupancy: 50%,   32 * 32 * 29 < 65,536, limiting factor: blocks
    c. blocks:  7, occupancy: 87.5%, 7 * 256 * 34 < 65,536, limiting factor: registers
```

9. A student mentions that they were able to multiply two 1024 x 1024 matrices using a matrix multiplication kernel with 32 x 32 thread blocks. The student is using a CUDA device that allows up to 512 threads per block and up to 8 blocks per SM. The student further mentions that each thread in a thread block calculates one element of the result matrix. What would be your reaction and why?

```python
    The student's approach to multiplying two 1024 x 1024 matrices using a matrix multiplication kernel with 32 x 32 thread blocks raises several concerns:

    1. Matrix Size and Thread Blocks
    Matrix Size: A 1024 x 1024 matrix has 1,048,576 elements.
    Thread Blocks: Each thread block has 32 × 32 = 1024 threads.
    2. Total Threads Needed
    To compute all elements of the resulting matrix, 1024×1024=1,048,576 threads are needed.
    3. Number of Blocks Required
    Given that each block contains 1024 threads, the number of blocks required for the entire operation is:
    Number of Blocks = 1,048,576 elements / 1024 threads/block = 1024 blocks

    4. CUDA Device Limits
    The student's CUDA device allows up to 512 threads per block and 8 blocks per SM. The use of 32 x 32 blocks is fine since it fits within the 512 threads per block limit, but the key issue lies in the number of blocks.
    5. Occupancy and Performance
    With 1024 blocks needed and only 8 blocks allowed per SM, the execution would require multiple kernel launches or would not achieve high occupancy. This could lead to poor performance due to underutilization of the device.
    Conclusion
    Reaction:

    Clarification Needed: I would clarify how the student managed to execute the multiplication with the constraints they mentioned. If they did manage to launch all necessary blocks, it likely involved multiple kernel launches or some form of tiling.
    Performance Impact: I would explain that while it may be possible to compute the result, the performance might not be optimal due to the limitations on the number of blocks per SM, which could result in significant overhead and inefficiencies.
    Overall, the student's understanding of kernel execution and the implications of hardware limits on performance should be reinforced.
```
