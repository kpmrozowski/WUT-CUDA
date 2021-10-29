# nvprop
make #&& ./l4z1
cuda-memcheck ./l4z1
nvprof ./l4z1 && make clean