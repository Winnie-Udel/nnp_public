#include <cuda.h> /* Header file for CUDA */

__device__ float deviceRelu(float x);
__device__ float deviceDRelu(float y);

__global__ void forwardKernel(
    float *input,
    float *weights,
    float *bias,
    float *output,
    int inputSize,
    int outputSize,
    int activation
);
__global__ void softMaxKernel(
    float *z, 
    float *out, 
    int len
);

__global__ void firstBackPropKernel(
    float *label,
    float *prediction, 
    float *output, 
    int classSize
);
__global__ void backPropKernel(
    float *currLayerOutput,
    float *nextWeights,
    float *nextDelta,
    float *output,
    int currLayerSize,
    int nextLayerSize
);

__global__ void updateWeightsKernel(
    float *outputWeights, 
    float *delta, 
    float *input, 
    float learningRate, 
    int inputSize, 
    int outputSize
);
__global__ void updateBiasKernel(
    float *outputBias,
    float *delta,
    float learningRate,
    int outputSize
);