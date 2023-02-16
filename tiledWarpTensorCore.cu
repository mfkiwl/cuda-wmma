/* Copyright (c) 1993-2017, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/*

 *  Compile
 *
 *  nvcc -o simpleTensorCoreMM simpleTensorCoreMM.cu -arch=sm_70 -lcurand
 *
*/

#include <stdio.h>
#include <curand.h>

// Define some error checking macros.
#define cudaErrCheck(stat) { cudaErrCheck_((stat), __FILE__, __LINE__); }
void cudaErrCheck_(cudaError_t stat, const char *file, int line) {
   if (stat != cudaSuccess) {
      fprintf(stderr, "CUDA Error: %s %s %d\n", cudaGetErrorString(stat), file, line);
   }
}

#define curandErrCheck(stat) { curandErrCheck_((stat), __FILE__, __LINE__); }
void curandErrCheck_(curandStatus_t stat, const char *file, int line) {
   if (stat != CURAND_STATUS_SUCCESS) {
      fprintf(stderr, "cuRand Error: %d %s %d\n", stat, file, line);
   }
}


#include <mma.h>
using namespace nvcuda;

// Must be multiples of 16 for wmma code to work
#define MATRIX_M 8192
#define MATRIX_N 8192
#define MATRIX_K 8192

#define TILE_WIDTH 16

// The only dimensions currently supported by WMMA
const int WMMA_M = 16;
const int WMMA_N = 16;
const int WMMA_K = 16;


// NUEVO KERNEL             ###################################################
// COMPLETE GEMM HIERARCHY  ###################################################

// ############################################################################
// ############################################################################
__global__ void wmma_example(half *a, half *b, float *c, int M, int N, int K, float alpha, float beta) {
  // create shorthand names for threadIdx & blockIdx
  int tx = threadIdx.x, ty = threadIdx.y;

  // Leading dimensions. Packed with no transpositions.
  int lda = M;
  int ldb = K;
  int ldc = M;

   // Tile using a 2D grid
   // Warps are numbered in two dimensions
   int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
   int warpN = (blockIdx.y * blockDim.y + threadIdx.y);

   // Declare the fragments
   wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> a_frag;
   wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;
   wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;
   wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

   wmma::fill_fragment(acc_frag, 0.0f);

   // allocate 2D tiles in __shared__ memory
   __shared__ half s_a[TILE_WIDTH][TILE_WIDTH];
   __shared__ half s_b[TILE_WIDTH][TILE_WIDTH];

   // Loop over k
   for (int i = 0; i < K; i += WMMA_K) {
      int aRow = warpM * WMMA_M;
      int aCol = i;

      int bRow = i;
      int bCol = warpN * WMMA_N;

      // collaboratively load tiles into __shared__
      s_a[ty][tx] = a[aRow*N + bRow];
      s_b[ty][tx] = b[bRow*N + bCol];

      // wait until all data is loaded before allowing
      // any thread in this block to continue
      __syncthreads();

      // Bounds checking
      if (aRow < M && aCol < K && bRow < K && bCol < N) {
         // Load the inputs
         wmma::load_matrix_sync(a_frag, *(s_a + aRow + aCol * lda), lda); //s_a + aRow + aCol * lda
         wmma::load_matrix_sync(b_frag, *(s_b + bRow + bCol * ldb), ldb); //s_b + bRow + bCol * ldb

         // Perform the matrix multiplication
         wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);

      }
   }

   // Load in the current value of c, scale it by beta, and add this our result scaled by alpha
   int cRow = warpM * WMMA_M;
   int cCol = warpN * WMMA_N;

   if (cRow < M && cCol < N) {
      wmma::load_matrix_sync(c_frag, c + cRow + cCol * ldc, ldc, wmma::mem_col_major);


      for(int i=0; i < c_frag.num_elements; i++) {
         c_frag.x[i] = alpha * acc_frag.x[i] + beta * c_frag.x[i];
      }

      // Store the output
      wmma::store_matrix_sync(c + cRow + cCol * ldc, c_frag, ldc, wmma::mem_col_major);
   }
}

// ############################################################################
// ############################################################################

// a simple version of matrix_multiply which issues redundant loads from off-chip global memory
__global__ void matrix_multiply_simple(half *a, half *b, float *ab, size_t width)
{
  // calculate the row & column index of the element
  int row = blockIdx.y*blockDim.y + threadIdx.y;
  int col = blockIdx.x*blockDim.x + threadIdx.x;

  float result = 0;

  // do dot product between row of a and column of b
  for(int k = 0; k < width; ++k)
  {
    result += (float )a[row*width+k] * (float )b[k*width+col];
  }

  // write out this thread's result
  ab[row*width+col] = result;
}


// an optimized version of matrix_multiplication which eliminates redundant loads
__global__ void matrix_multiply(half *a, half *b, float *ab, size_t width)
{
  // create shorthand names for threadIdx & blockIdx
  int tx = threadIdx.x, ty = threadIdx.y;
  int bx = blockIdx.x,  by = blockIdx.y;

  // allocate 2D tiles in __shared__ memory
  __shared__ float s_a[TILE_WIDTH][TILE_WIDTH];
  __shared__ float s_b[TILE_WIDTH][TILE_WIDTH];

  // calculate the row & column index of the element
  int row = by*blockDim.y + ty;
  int col = bx*blockDim.x + tx;

  float result = 0;

  // loop over the tiles of the input in phases
  for(int p = 0; p < width/TILE_WIDTH; ++p)
  {
    // collaboratively load tiles into __shared__
    s_a[ty][tx] = a[row*width + (p*TILE_WIDTH + tx)];
    s_b[ty][tx] = b[(p*TILE_WIDTH + ty)*width + col];

    // wait until all data is loaded before allowing
    // any thread in this block to continue
    __syncthreads();

    // do dot product between row of s_a and column of s_b
    for(int k = 0; k < TILE_WIDTH; ++k)
    {
      result += s_a[ty][k] * s_b[k][tx];
    }

    // wait until all threads are finished with the data
    // before allowing any thread in this block to continue
    __syncthreads();
  }

  // write out this thread's result
  ab[row*width+col] = result;
}




__global__ void convertFp32ToFp16 (half *out, float *in, int n) {
   int idx = blockDim.x * blockIdx.x + threadIdx.x;
   if (idx < n) {
      out[idx] = in[idx];
   }
}

int main(int argc, char* argv[]) {
   float *a_fp32;
   float *b_fp32;
   half *a_fp16;
   half *b_fp16;

   float *c;
   float *c_wmma;

   float *c_host_wmma;

   curandGenerator_t gen;

   const dim3 block_size(TILE_WIDTH,TILE_WIDTH);
   const dim3 num_blocks(MATRIX_M/block_size.x, MATRIX_N/block_size.y);


   cudaErrCheck(cudaMalloc((void**)&a_fp32, MATRIX_M * MATRIX_K * sizeof(float)));
   cudaErrCheck(cudaMalloc((void**)&b_fp32, MATRIX_K * MATRIX_N * sizeof(float)));
   cudaErrCheck(cudaMalloc((void**)&a_fp16, MATRIX_M * MATRIX_K * sizeof(half)));
   cudaErrCheck(cudaMalloc((void**)&b_fp16, MATRIX_K * MATRIX_N * sizeof(half)));

   cudaErrCheck(cudaMalloc((void**)&c, MATRIX_M * MATRIX_N * sizeof(float)));
   cudaErrCheck(cudaMalloc((void**)&c_wmma, MATRIX_M * MATRIX_N * sizeof(float)));

   c_host_wmma = (float*)malloc(MATRIX_M * MATRIX_N * sizeof(float));

   curandErrCheck(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
   curandErrCheck(curandSetPseudoRandomGeneratorSeed(gen, 1337ULL));

   curandErrCheck(curandGenerateUniform(gen, a_fp32, MATRIX_M * MATRIX_K));
   curandErrCheck(curandGenerateUniform(gen, b_fp32, MATRIX_K * MATRIX_N));

   // curand doesn't currently support fp16 so we generate in fp32 and convert to fp16.
   convertFp32ToFp16 <<< (MATRIX_M * MATRIX_K + 255) / 256, 256 >>> (a_fp16, a_fp32, MATRIX_M * MATRIX_K);
   convertFp32ToFp16 <<< (MATRIX_K * MATRIX_N + 255) / 256, 256 >>> (b_fp16, b_fp32, MATRIX_K * MATRIX_N);

   curandErrCheck(curandGenerateUniform(gen, c, MATRIX_M * MATRIX_N));

   curandErrCheck(curandDestroyGenerator(gen));

   cudaErrCheck(cudaMemcpy(c_wmma, c, MATRIX_M * MATRIX_N * sizeof(float), cudaMemcpyDeviceToDevice));


   // time the kernel launches using CUDA events
   cudaEvent_t launch_begin, launch_end;
   cudaEventCreate(&launch_begin);
   cudaEventCreate(&launch_end);

   // PART 1. TESTING SIMPLE KERNEL
   // to get accurate timings, launch a single "warm-up" kernel
   matrix_multiply_simple<<<num_blocks,block_size>>>(a_fp16, b_fp16, c_wmma, MATRIX_M );

   // time many kernel launches and take the average time
   const size_t num_launches = 10;
   float average_simple_time = 0;
   printf("Timing simple implementation...\n");

   for(int i = 0; i < num_launches; ++i)
   {
     // record a CUDA event immediately before and after the kernel launch
     cudaEventRecord(launch_begin,0);
     matrix_multiply_simple<<<num_blocks,block_size>>>(a_fp16, b_fp16, c_wmma, MATRIX_M );
     cudaEventRecord(launch_end,0);
     cudaEventSynchronize(launch_end);

     // measure the time spent in the kernel
     float time = 0;
     cudaEventElapsedTime(&time, launch_begin, launch_end);

     average_simple_time += time;

     printf( "Test %3d, Time %f\n", i, time );
   }
   average_simple_time /= num_launches;
   printf( "Done.\n" );

   // PART 2. TESTING OPTIMIZED KERNEL
   // now time the optimized kernel
   // again, launch a single "warm-up" kernel
   matrix_multiply<<<num_blocks,block_size>>>(a_fp16, b_fp16, c_wmma, MATRIX_M );

   // time many kernel launches and take the average time
   float average_optimized_time = 0;
   printf("Timing optimized implementation...\n");
   for(int i = 0; i < num_launches; ++i)
   {
     // record a CUDA event immediately before and after the kernel launch
     cudaEventRecord(launch_begin,0);
     matrix_multiply<<<num_blocks,block_size>>>(a_fp16, b_fp16, c_wmma, MATRIX_M );
     cudaEventRecord(launch_end,0);
     cudaEventSynchronize(launch_end);

     // measure the time spent in the kernel
     float time = 0;
     cudaEventElapsedTime(&time, launch_begin, launch_end);

     average_optimized_time += time;
     printf( "Test %3d, Time %f\n", i, time );
   }
   average_optimized_time /= num_launches;
   printf( "Done.\n" );

   // PART 3. TESTING KERNEL USING TENSOR CORES
   float alpha = 2.0f;
   float beta = 2.0f;


   printf("\nM = %d, N = %d, K = %d. alpha = %f, beta = %f\n\n", MATRIX_M, MATRIX_N, MATRIX_K, alpha, beta);

   // First: using WMMA
   dim3 gridDim;
   dim3 blockDim;

   // blockDim.x must be a multple of warpSize
   // 128x4 means we have 16 warps and a block computes a 64x64 output tile
   blockDim.x = 128;
   blockDim.y = 4;

   gridDim.x = (MATRIX_M + (WMMA_M * blockDim.x / 32 - 1)) / (WMMA_M * blockDim.x / 32);
   gridDim.y = (MATRIX_N + WMMA_N * blockDim.y - 1) / (WMMA_N * blockDim.y);

   printf("Running with wmma with grid (%d,%d), block(%d,%d)...\n", gridDim.x, gridDim.y, blockDim.x, blockDim.y );
   wmma_example <<< gridDim, blockDim >>> (a_fp16, b_fp16, c_wmma, MATRIX_M, MATRIX_N, MATRIX_K, alpha, beta);

   // time many kernel launches and take the average time
   float average_tensor_time = 0;
   printf("Timing kernel using tensor cores...\n");
   for(int i = 0; i < num_launches; ++i)
   {
     // record a CUDA event immediately before and after the kernel launch
     cudaEventRecord(launch_begin,0);
     wmma_example <<< gridDim, blockDim >>> (a_fp16, b_fp16, c_wmma, MATRIX_M, MATRIX_N, MATRIX_K, alpha, beta);
     cudaEventRecord(launch_end,0);
     cudaEventSynchronize(launch_end);

     // measure the time spent in the kernel
     float time = 0;
     cudaEventElapsedTime(&time, launch_begin, launch_end);

     average_tensor_time += time;
     printf( "Test %3d, Time %f\n", i, time );
   }
   average_tensor_time /= num_launches;
   printf( "Done.\n" );

   // For error checking
   // cudaErrCheck(cudaMemcpy(c_host_wmma, c_wmma, MATRIX_M * MATRIX_N * sizeof(float), cudaMemcpyDeviceToHost));

   // report the effective throughput of each kernel in TFLOPS
   // the effective throughput is measured as the number of floating point operations performed per second:
   // (one mul + one add) * N^3
   float simple_throughput = (2.0f * (float ) MATRIX_M * (float )MATRIX_N * (float )MATRIX_K) / (average_simple_time / 1000.0f) / 1000000000.0f;
   simple_throughput /= 1000;   // TFLOPS
   float optimized_throughput = (2.0f * (float ) MATRIX_M * (float )MATRIX_N * (float )MATRIX_K) / (average_optimized_time / 1000.0f) / 1000000000.0f;
   optimized_throughput /= 1000;   // TFLOPS
   float wmma_th = (2.0f * (float ) MATRIX_M * (float )MATRIX_N * (float )MATRIX_K) / (average_tensor_time / 1000.0f) / 1000000000.0f;
   wmma_th /= 1000;   // TFLOPS

   printf("Simple kernel average time  %fms, %f TFLOPS\n", average_simple_time, simple_throughput );
   printf("Optimized kernel average time  %fms, %f TFLOPS\n", average_optimized_time, optimized_throughput );
   printf("Kernel using Complete GEMM Hierarchy time  %fms, %f TFLOPS\n", average_tensor_time, wmma_th );

   cudaErrCheck(cudaEventDestroy(launch_begin));
   cudaErrCheck(cudaEventDestroy(launch_end));

   cudaErrCheck(cudaFree(a_fp32));
   cudaErrCheck(cudaFree(b_fp32));
   cudaErrCheck(cudaFree(a_fp16));
   cudaErrCheck(cudaFree(b_fp16));

   cudaErrCheck(cudaFree(c));
   cudaErrCheck(cudaFree(c_wmma));

   free(c_host_wmma);

   cudaErrCheck(cudaDeviceReset());
   return 0;
}
