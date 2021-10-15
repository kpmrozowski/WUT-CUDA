#include <iostream>

/** 
@see https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
*/
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
} // 208 mic sek

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
} // 208 mic sek

//shared memory jest zorganizowana w 32 banki, w zaleznosci od precyzji moga byc wieksze lub mniejsze. Standardowe maja 32 bity. Konflikty w odczytywaniu tych bankow to Bank Conflicts. Elementy tabeli sa kazdy w innym banku, ale co 32 jest w tym samym, Odczytywanie z tego samego banku to konflikt. 

__global__ void reduce2(int *in, int *out) {
    extern __shared__ int shm[]; // pamiec dzielona
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    shm[tid] = in[gid];
    __syncthreads();

    for (unsigned int stride = blockDim.x / 2; stride > 0 ; stride /= 2) {
        if (tid < stride) {
            shm[tid] += shm[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        out[blockIdx.x] = shm[0];
    }
} // 208 mic sek

__global__ void reduce3(int *in, int *out) {
    extern __shared__ int shm[]; // pamiec dzielona
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    shm[tid] = in[gid] + in[gid + blockDim.x];
    __syncthreads();

    for (unsigned int stride = blockDim.x / 2; stride > 0 ; stride /= 2) {
        if (tid < stride) {
            shm[tid] += shm[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        out[blockIdx.x] = shm[0];
    }
} // 210.94us

// 
// bez volatile jest optymalizacja i nie dziala
__device__ void warpReduce(volatile int *data, int tid) {
    data[tid] += data[tid + 32];
    data[tid] += data[tid + 16];
    data[tid] += data[tid + 8];
    data[tid] += data[tid + 4];
    data[tid] += data[tid + 2];
    data[tid] += data[tid + 1];
}

__global__ void reduce4(int *in, int *out) {
    extern __shared__ int shm[]; // pamiec dzielona
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    shm[tid] = in[gid] + in[gid + blockDim.x];
    __syncthreads();

    for (unsigned int stride = blockDim.x / 2; stride > 32 ; stride /= 2) {
        if (tid < stride) {
            shm[tid] += shm[tid + stride];
        }
        __syncthreads();
    }

    if (tid < 32) {
        warpReduce(shm, tid);
    }

    if (tid == 0) {
        out[blockIdx.x] = shm[0];
    }
} // 72.192us

template <unsigned int blockSize>
__device__ void warpReduce(volatile int* sdata, int tid) {
    if (blockSize >= 512) {
        if (tid < 256) { 
            sdata[tid] += sdata[tid + 256]; 
        } __syncthreads(); 
    }
    if (blockSize >= 256) {
        if (tid < 128) { 
            sdata[tid] += sdata[tid + 128]; 
        } __syncthreads(); 
    }
    if (blockSize >= 128) {
        if (tid <  64)  { 
            sdata[tid] += sdata[tid +   64]; 
        } __syncthreads(); 
    }
    if (tid < 32) warpReduce<blockSize>(sdata, tid);
    if (blockSize >= 64) sdata[tid] += sdata[tid + 32]; 
    if (blockSize >= 32) sdata[tid] += sdata[tid + 16]; 
    if (blockSize >= 16) sdata[tid] += sdata[tid +  8]; 
    if (blockSize >=   8) sdata[tid] += sdata[tid +  4]; 
    if (blockSize >=   4) sdata[tid] += sdata[tid +  2]; 
    if (blockSize >=   2) sdata[tid] += sdata[tid +  1];
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
    // int reduce_num = 5;
    int num_threads = 1024;

    // switch (reduce_num) {
    //     case 0: {
    //         int num_blocks = N / 1024;
    //         int shm_size = num_threads * sizeof(int);
    //         reduce0<<<num_blocks, num_threads, shm_size>>>(d_in, d_out);
    //         reduce0<<<1, num_threads, shm_size>>>(d_out, d_out);
    //     }
    //     case 1: {
    //         int num_blocks = N / 1024;
    //         int shm_size = num_threads * sizeof(int);
    //         reduce1<<<num_blocks, num_threads, shm_size>>>(d_in, d_out);
    //         reduce1<<<1, num_threads, shm_size>>>(d_out, d_out);
    //     }
    //     case 2: {
    //         int num_blocks = N / 1024;
    //         int shm_size = num_threads * sizeof(int);
    //         reduce2<<<num_blocks, num_threads, shm_size>>>(d_in, d_out);
    //         reduce2<<<1, num_threads, shm_size>>>(d_out, d_out);
    //     }
    //     case 3: {
    //         int num_blocks = N / 1024 / 2;
    //         int shm_size = num_threads * sizeof(int);
    //         reduce3<<<num_blocks, num_threads, shm_size>>>(d_in, d_out);
    //         reduce3<<<1, num_threads, shm_size>>>(d_out, d_out);
    //     }
    //     case 4: {
    //         int num_blocks = N / 1024 / 2;
    //         int shm_size = num_threads * sizeof(int);
    //         reduce4<<<num_blocks, num_threads, shm_size>>>(d_in, d_out);
    //         reduce4<<<1, num_threads, shm_size>>>(d_out, d_out);
    //     }
    // }
    int num_blocks = N / 1024 / 2;
    int shm_size = num_threads * sizeof(int);
    reduce4<<<num_blocks, num_threads, shm_size>>>(d_in, d_out);
    reduce4<<<1, num_threads, shm_size>>>(d_out, d_out);

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