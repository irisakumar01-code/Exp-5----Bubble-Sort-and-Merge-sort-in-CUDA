# Exp5 Bubble Sort and Merge sort in CUDA
**Objective:**
Implement Bubble Sort and Merge Sort on the GPU using CUDA, analyze the efficiency of this sorting algorithm when parallelized, and explore the limitations of Bubble Sort and Merge Sort for large datasets.
## AIM:
Implement Bubble Sort and Merge Sort on the GPU using CUDA to enhance the performance of sorting tasks by parallelizing comparisons and swaps within the sorting algorithm.

Code Overview:
You will work with the provided CUDA implementation of Bubble Sort and Merge Sort. The code initializes an unsorted array, applies the Bubble Sort, Merge Sort algorithm in parallel on the GPU, and returns the sorted array as output.

## EQUIPMENTS REQUIRED:
Hardware – PCs with NVIDIA GPU & CUDA NVCC, Google Colab with NVCC Compiler, CUDA Toolkit installed, and sample datasets for testing.

## PROCEDURE:

Tasks:

a. Modify the Kernel:

Implement Bubble Sort and Merge Sort using CUDA by assigning each comparison and swap task to individual threads.
Ensure the kernel checks boundaries to avoid out-of-bounds access, particularly for edge cases.
b. Performance Analysis:

Measure the execution time of the CUDA Bubble Sort with different array sizes (e.g., 512, 1024, 2048 elements).
Experiment with various block sizes (e.g., 16, 32, 64 threads per block) to analyze their effect on execution time and efficiency.
c. Comparison:

Compare the performance of the CUDA-based Bubble Sort and Merge Sort with a CPU-based Bubble Sort and Merge Sort implementation.
Discuss the differences in execution time and explain the limitations of Bubble Sort and Merge Sort when parallelized on the GPU.
## PROGRAM:
~~~
%%writefile sorting.cu

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <chrono>
#include <algorithm>
#include <vector>

using namespace std;

// GPU Bubble Sort Kernel
__global__ void bubbleSortKernel(int *arr, int n, int phase)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int left = 2 * idx + phase;

    if (left + 1 < n)
    {
        if (arr[left] > arr[left + 1])
        {
            int temp = arr[left];
            arr[left] = arr[left + 1];
            arr[left + 1] = temp;
        }
    }
}

// GPU Merge Sort Kernel
__global__ void mergeSortKernel(int *arr, int *temp, int n, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int left = idx * 2 * size;

    if (left >= n)
        return;

    int mid = min(left + size, n);
    int right = min(left + 2 * size, n);

    int i = left;
    int j = mid;
    int k = left;

    while (i < mid && j < right)
    {
        if (arr[i] <= arr[j])
            temp[k++] = arr[i++];
        else
            temp[k++] = arr[j++];
    }

    while (i < mid)
        temp[k++] = arr[i++];

    while (j < right)
        temp[k++] = arr[j++];
}

// Host Merge Function
void mergeHost(int *arr, int left, int mid, int right)
{
    int n1 = mid - left + 1;
    int n2 = right - mid;

    int *L = (int*)malloc(n1 * sizeof(int));
    int *R = (int*)malloc(n2 * sizeof(int));

    for (int i = 0; i < n1; i++)
        L[i] = arr[left + i];

    for (int j = 0; j < n2; j++)
        R[j] = arr[mid + 1 + j];

    int i = 0, j = 0, k = left;

    while (i < n1 && j < n2)
    {
        if (L[i] <= R[j])
            arr[k++] = L[i++];
        else
            arr[k++] = R[j++];
    }

    while (i < n1)
        arr[k++] = L[i++];

    while (j < n2)
        arr[k++] = R[j++];

    free(L);
    free(R);
}

// Bubble Sort on CPU
void bubbleSortCPU(int *arr, int n)
{
    auto start = chrono::high_resolution_clock::now();

    for (int i = 0; i < n - 1; i++)
    {
        for (int j = 0; j < n - i - 1; j++)
        {
            if (arr[j] > arr[j + 1])
            {
                int temp = arr[j];
                arr[j] = arr[j + 1];
                arr[j + 1] = temp;
            }
        }
    }

    auto end = chrono::high_resolution_clock::now();

    double time = chrono::duration<double, milli>(end - start).count();

    printf("Bubble Sort (CPU) took %f milliseconds\n", time);
}

// Merge Sort on CPU
void mergeSortCPU(int *arr, int n)
{
    auto start = chrono::high_resolution_clock::now();

    for (int size = 1; size < n; size *= 2)
    {
        for (int left = 0; left < n; left += 2 * size)
        {
            int mid = min(left + size, n);
            int right = min(left + 2 * size, n);

            if (mid < right)
                mergeHost(arr, left, mid - 1, right - 1);
        }
    }

    auto end = chrono::high_resolution_clock::now();

    double time = chrono::duration<double, milli>(end - start).count();

    printf("Merge Sort (CPU) took %f milliseconds\n", time);
}

// Bubble Sort on GPU
void bubbleSortGPU(int *arr, int n)
{
    int *d_arr;

    cudaMalloc(&d_arr, n * sizeof(int));

    cudaMemcpy(d_arr, arr, n * sizeof(int), cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = ((n + 1) / 2 + threads - 1) / threads;

    cudaEvent_t start, stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    for (int phase = 0; phase < n; phase++)
    {
        bubbleSortKernel<<<blocks, threads>>>(d_arr, n, phase % 2);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds;

    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaMemcpy(arr, d_arr, n * sizeof(int), cudaMemcpyDeviceToHost);

    printf("Bubble Sort (GPU) took %f milliseconds\n", milliseconds);

    cudaFree(d_arr);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

// Merge Sort on GPU
void mergeSortGPU(int *arr, int n)
{
    int *d_arr, *d_temp;

    cudaMalloc(&d_arr, n * sizeof(int));
    cudaMalloc(&d_temp, n * sizeof(int));

    cudaMemcpy(d_arr, arr, n * sizeof(int), cudaMemcpyHostToDevice);

    int threads = 256;

    int *src = d_arr;
    int *dst = d_temp;

    cudaEvent_t start, stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    for (int size = 1; size < n; size *= 2)
    {
        int segments = (n + 2 * size - 1) / (2 * size);

        int blocks = (segments + threads - 1) / threads;

        mergeSortKernel<<<blocks, threads>>>(src, dst, n, size);

        int *temp = src;
        src = dst;
        dst = temp;
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds;

    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaMemcpy(arr, src, n * sizeof(int), cudaMemcpyDeviceToHost);

    printf("Merge Sort (GPU) took %f milliseconds\n", milliseconds);

    cudaFree(d_arr);
    cudaFree(d_temp);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

// Print Array
void printArray(int *arr, int n)
{
    for (int i = 0; i < n; i++)
        printf("%d ", arr[i]);

    printf("\n");
}

// Main Function
int main()
{
    int n = 10000;

    int *arr = (int*)malloc(n * sizeof(int));

    // Generate random array
    for (int i = 0; i < n; i++)
        arr[i] = rand() % 1000;

    printf("Original array: \n");
    printArray(arr, n);

    // Bubble Sort CPU
    bubbleSortCPU(arr, n);

    printf("Sorted array using Bubble Sort (CPU): \n");
    printArray(arr, n);

    // Generate random array again
    for (int i = 0; i < n; i++)
        arr[i] = rand() % 1000;

    // Bubble Sort GPU
    bubbleSortGPU(arr, n);

    printf("Sorted array using Bubble Sort (GPU): \n");
    printArray(arr, n);

    // Generate random array again
    for (int i = 0; i < n; i++)
        arr[i] = rand() % 1000;

    printf("Original array: \n");
    printArray(arr, n);

    // Merge Sort CPU
    mergeSortCPU(arr, n);

    printf("Sorted array using Merge Sort (CPU): \n");
    printArray(arr, n);

    // Generate random array again
    for (int i = 0; i < n; i++)
        arr[i] = rand() % 1000;

    // Merge Sort GPU
    mergeSortGPU(arr, n);

    printf("Sorted array using Merge Sort (GPU): \n");
    printArray(arr, n);

    free(arr);

    return 0;
}
~~~


## OUTPUT:
<img width="1700" height="460" alt="image" src="https://github.com/user-attachments/assets/74f2db4b-d77c-4d2c-8126-403945e9846e" />



## RESULT:
Thus, the program has been executed using CUDA to implement and compare the execution time of Bubble Sort and Merge Sort algorithms on both CPU and GPU.
