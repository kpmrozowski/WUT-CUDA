#include <iostream>

__global__ void reduce0(int *in, int *out) {
    extern __shared__ int shm[]; // pamiec dzielona
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    shm[tid] = in[gid];
    __syncthreads();

    for (unsigned int stride = 1; stride < blockDim.x; stride *= 2) {
        if (tid % (2 * stride) == 0) {
            shm[tid] += shm[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        out[blockIdx.x] = shm[0];
    }
}

__global__ void reduce1(int *in, int *out) {
    extern __shared__ int shm[]; // pamiec dzielona
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    shm[tid] = in[gid];
    __syncthreads();

    for (unsigned int stride = 1; stride < blockDim.x; stride *= 2) {
        unsigned int index = 2 * stride * tid;
        if (index < blockDim.x) {
            shm[index] += shm[index + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        out[blockIdx.x] = shm[0];
    }
}

int main(int argc, char **argv) {
    const int N = 1 << 20;
    int *h_data = new int[N];
    int *d_in, *d_out;

    cudaMalloc(&d_in, N * sizeof(int));
    cudaMalloc(&d_out, N * sizeof(int));

    cudaMemset(d_out, 0, N * sizeof(int));

    for (int i = 0; i < N; ++i) {
        h_data[i] = 1;
    }

    cudaMemcpy(d_in, h_data, N * sizeof(int), cudaMemcpyHostToDevice);

    // reduce ....
    int num_threads = 1024;
    int num_blocks = N / 1024;
    int shm_size = num_threads * sizeof(int);
    reduce1<<<num_blocks, num_threads, shm_size>>>(d_in, d_out);
    reduce1<<<1, num_threads, shm_size>>>(d_out, d_out);

    cudaMemcpy(h_data, d_out, N * sizeof(int), cudaMemcpyDeviceToHost);

    if (h_data[0] != N) {
        std::cout << "incorrect result" << std::endl;
        return 1;
    }

    delete[] h_data;
    cudaFree(d_in);
    cudaFree(d_out);
    return 0;
}