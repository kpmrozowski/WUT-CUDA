# nvprop
make && ./reduce
cuda-memcheck ./reduce
nvprof ./reduce && make clean