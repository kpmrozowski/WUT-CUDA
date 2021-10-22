# nvprop
nvcc fractalGL.cu -o fractalGL -I/opt/cuda/samples/common/inc -lGL -lGLU -lglut && ./fractalGL -9 -10 -1 10 50 1000
# cuda-memcheck ./fractalGL
# nvprof ./fractalGL && make clean