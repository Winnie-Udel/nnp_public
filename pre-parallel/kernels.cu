#include "kernels.h"
#include <math.h>

/**
 * Device activation functions for ReLU layers 
 * 
 * Arguments: 
 *  x: input value 
 * 
 * Returns: 
 *  activated value based on ReLU function
 */
__device__ float deviceRelu(float x) {
    return x > 0 ? x : 0;
}

/**
 * Device derivative of ReLU activation function 
 * 
 * Arguments:
 *  y: output value from ReLU function
 * 
 * Returns:
 *  Derivative value
 */
__device__ float deviceDRelu(float y) {
    return y > 0 ? 1 : 0;
}

/**
 * CUDA kernel to perform a forward pass, which includes:
 *  - Matrix-vector multiplication 
 *  - Adding bias term
 *  - Applying activation function when specified 
 * 
 * Each thread computes one output feature: 
 *  - output[j] = activation(bias[j] + Σ(input[i] * weights[i][j]))
 *  - Otherwise can be seen as y = activation(b + Wx), where b is bias vectors, x is 
 *    input vector, and W is the weight matrix
 * 
 * Arguments: 
 *  input: pointer to the input vector (or trained data) 
 *  weights: pointer to the weight matrix
 *  bias: pointer to the bias vector
 *  output: pointer to output vector 
 *  inputSize: number of input features
 *  outputSize: number of output features
 *  activation: 0 if no activation, 1 if relu activation
 * 
*/
__global__ void forwardKernel(
    float *input,
    float *weights,
    float *bias,
    float *output,
    int inputSize,
    int outputSize,
    int activation
) {
    // Shared memory for input vector, dynamic allocation 
    extern __shared__ float sharedInput[];

    int tId = threadIdx.x;
    // Index for output feature 
    // Independent computation with multiple blocks
    int j = blockIdx.x * blockDim.x + threadIdx.x; 

    for (int i = tId; i < inputSize; i+= blockDim.x){
        sharedInput[i] = input[i]
    }
    __syncthreads();

    if (j < outputSize) {
        // Initialize with bias
        float sum = bias[j];

        // Matrix-vector multiplication
        for (int i = 0; i < inputSize; i++) {
            sum += sharedInput[i] * weights[i * outputSize + j];
        }

        // Apply activation function 
        if (activation == 1) {
            sum = deviceRelu(sum);
        }

        // Write result to output
        output[j] = sum;
    }
}

/**
 * CUDA kernel to perform softmax.
 * 
 * Arguments: 
 *  z: pointer to input array 
 *  out: pointer to output array that stores the softmax results
 *  len: length of the input/output arrays 
*/
__global__ void softMaxKernel(
    float *z, 
    float *out, 
    int len
) {
    // Index for output 
    // This is the thread index in its own block
    int j = threadIdx.x;

    if (j < len) {
        // Finding the max value
        float max = z[0];
        for (int i = 1; i < len; i++) {
            if (z[i] > max) max = z[i];
        }

        // Compute the exponent of this indexed value
        float exp = expf(z[j] - max);

        // Compute the sum 
        float sum = 0;
        for (int i = 0; i < len; i++) {
            sum += expf(z[i] - max);
        }

        // Normalize
        out[j] = exp / sum;
    }
}

/**
 * CUDA kernel to compute the first backproprogation. This determines
 * how wrong the output layer is by comparing the prediction against the true 
 * labels. 
 * 
 * Each thread computes one element of the output delta vector. 
 * 
 * Arguments:
 *  label: pointer to the label vector
 *  prediction: pointer to the network prediction vector
 *  output: pointer to the output delta vector
 *  classSize: number of classes (for this classification task)
 */
__global__ void firstBackPropKernel(
    float *label,
    float *prediction, 
    float *output, 
    int classSize
){
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (j < classSize){
        output[j] = label[j] - prediction[j];
    }
}

/**
 * CUDA kernel that computes backpropagation from the next layer into current layer.
 * It determines how much current layer causes the next layer's error. 
 * 
 * Each thread computes one element of the current delta vector. 
 * 
 * Arguments:
 *  currLayerOutput: pointer to output of current layer after ReLU
 *  nextWeights: pointer to the weight matrix (it is in the form of a vector though) of next layer
 *  nextDelta: pointer to the delta vector of the next layer
 *  output: pointer the the delta vector of the current layer
 *  currLayerSize: size of the current layer
 *  nextLayerSize: size of the next layer
 */
__global__ void backPropKernel(
    float *currLayerOutput, // Post-ReLU
    float *nextWeights,
    float *nextDelta,
    float *output,
    int currLayerSize,
    int nextLayerSize
){
    // Shared memory for the nextDelta vector, dynamic allocation
    extern __shared__ float sharedNextDelta[];
    
    int tId = threadIdx.x;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = tId; i < nextLayerSize ; i+= blockDim.x){
        sharedNextDelta[i] = nextDelta[i];
    }
    __syncthreads();

    if (j < currLayerSize){
        float err = 0;

        for (int k = 0; k < nextLayerSize; k++){
            err += sharedNextDelta[k]*nextWeights[j * nextLayerSize + k];
        }

        output[j] = err * deviceDRelu(currLayerOutput[j]);
    }
}

/**
 * CUDA kernel to update the weight matrix. 
 * 
 * Each thread update a single weight where j is the input index (row) and k is 
 * the output index (column).
 * 
 * Arguments:
 *  outputWeights: pointer to the updated weight matrix 
 *  delta: pointer to the delta vector of next layer
 *  input: pointer to the input vector
 *  learningRate: learning rate
 *  inputSize: size of input (rows)
 *  outputSize: size of output (columns)
 */
__global__ void updateWeightsKernel(
    float *outputWeights, 
    float *delta, 
    float *input, 
    float learningRate, 
    int inputSize, 
    int outputSize
){

    // Index of input (row of weight matrix) 
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    // Index of output (column of weight matrix)
    int k = blockIdx.y * blockDim.y + threadIdx.y;

    if (j < inputSize && k < outputSize){
        // Update weight
        outputWeights[j * outputSize + k] += learningRate*delta[k]*input[j];
    }
}


/**
 * CUDA kernel to update the bias vector.
 * 
 * Each thread updates one bias value.
 * 
 * Arguments:
 *  outputBias: pointer to the bias vector for the layer
 *  delta: pointer to the delta vector for the layer
 *  learningRate: learning rate
 *  outputSize: size of output
 */
__global__ void updateBiasKernel(
    float *outputBias,
    float *delta,
    float learningRate,
    int outputSize
){
    int k = blockIdx.x * blockDim.x + threadIdx.x;

    if (k < outputSize){
        outputBias[k] += learningRate * delta[k];
    }
}