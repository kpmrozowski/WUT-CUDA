#include <iostream>
#include <algorithm>

// SPMD -- single program multiple data
__global__ void add(float* v1, float* v2, int N) {
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x; // one thread 
    if (thread_id < N) {
        v1[thread_id] = v1[thread_id] + v2[thread_id];
    }
}

int main(int argc, char* argv[]) {
    const int N = (1 << 20);
    // std::array<float, N> vec1(1.0f), vec2(2.0f);
    float* h_vec1 = new float[N];
    float* h_vec2 = new float[N];
    float* d_vec1;
    float* d_vec2;
    cudaMalloc(&d_vec1, N * sizeof(float));
    cudaMalloc(&d_vec2, N * sizeof(float));
    for (int i = 0; i < N; ++i) {
        h_vec1[i] = 1.0f;
        h_vec2[i] = 2.0f;
    }

    cudaMemcpy(d_vec1, h_vec1, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vec2, h_vec2, N * sizeof(float), cudaMemcpyHostToDevice);

    // cudaMallocMenaged(&h_vec1, N * sizeof(float));
    // cudaMallocMenaged(&h_vec1, N * sizeof(float));

    dim3 num_threads(1024);
    dim3 num_blocks(N / 1024 + 1);

    add<<<num_blocks, num_threads>>>(d_vec1, d_vec2, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_vec1, d_vec1, N * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < N; ++i) {
        if (h_vec1[i] != 3) {
            std::cout << "Incorrect result!" << std::endl;
            return 1;
        }
    }
    delete[] h_vec1;
    delete[] h_vec2;
    cudaFree(d_vec1);
    cudaFree(d_vec2);
}
