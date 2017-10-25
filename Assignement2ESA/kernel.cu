//Set to 0 for added slow serial execution, 1 for only fast execution (debugging)
#define DEBUG 0

#define iterations		1000
#define N_mat			13 // mat size
#define MAX_acc			14 // extra thread for addition of non matrix mults
#define threads_per_block 4

// System includes
#include <stdio.h>
#include <assert.h>

// CUDA runtime
#include <cuda.h>
#include <cuda_runtime.h>

// Helper functions and utilities to work with CUDA
#include <helper_functions.h>
#include <helper_cuda.h>

//#include <device_atomic_functions.hpp>

__global__ void matrixMulCUDASlow(int *A, int *B, int *C) {

	for (uint8_t i = 0; i < N_mat; i++) {
		// ROW OPERATIONS
		for (uint8_t j = 0; j < N_mat; j++) {
			// COLUMN OPERATIONS
			int32_t Sum = 0;
			// CALCULATE DOT PRODUCT
			for (uint8_t k = 0; k < N_mat; k++) {
				Sum += A[(i * N_mat) + k] * B[(k * N_mat) + j];
			}
			C[(i * N_mat) + j] = Sum + A[(i * N_mat) + j] + B[(i * N_mat) + j];
		}
	}
}
__global__ void matrixMulCUDA(int *A, int *B, int *C)
{
	// Thread index
	int row = threadIdx.x + threads_per_block*blockIdx.x; // offset row with block
	int col = threadIdx.y;
	int	z = threadIdx.z;
	int index = (row * N_mat) + col;
	int offset = 2 * z;

	int multi[N_mat*N_mat];
	if (z == 0) multi[index] = 0;
	__syncthreads();

	if (row < 13) {// prevent rows larger than 12 to write, as these don't exist
		if (z == MAX_acc) { // extra thread
			atomicAdd(&multi[index], A[index] + B[index]);
		}
		else {
			atomicAdd(&multi[index], A[(row * N_mat) + offset] * B[col + (N_mat * offset)]);
		}
	}
	__syncthreads();
	if (z == MAX_acc && row < 13) { // let one thread push
		atomicExch(&C[index], multi[index]);
	}
}

void printmatrix(int m[N_mat*N_mat]) {
	for (size_t i = 0; i < N_mat; i++) {
		for (size_t j = 0; j < N_mat; j++) {
			std::cout << m[(N_mat * i) + j];
			std::cout << ",";
		}
		std::cout << "\n";
	}
}


int main()
{
	int a[N_mat*N_mat] = {
		28, 122,  80,   42,   54,   122,  98,   42,   99,   58,   124,  29,   21 ,
		113,  85,   30,   35,   41,   98,   103,  68,   15,   50,   31,   80,   54,
		47, 37,   23,   96,   59,   47,   84,   26,   84,   72,   51,   118,  119 ,
		38, 121,  45,   21,   87,   91,   20,   69,   98,   119,  15,   89,   47 ,
		40, 71,   105,  76,   31,   65,   109,  30,   127,  110,  17,   64,   64 ,
		45, 20,   113,  86,   86,   51,   104,  115,  61,   103,  60,   113,  44 ,
		101,  107,  33,   63,   39,   47,   120,  20,   41,   64,   102,  59,   86 ,
		9,  42,   118,  26,   83,   123,  10,   82,   47,   108,  127,  4,    66 ,
		75, 26,   117,  80,   47,   111,  38,   22,   98,   101,  92,   100,  48 ,
		90, 9,    71,   36,   90,   95,   4,    94,   72,   29,   77,   118,  78 ,
		81, 75,   97,   127,  22,   8,    96,   80,   100,  88,   69,   114,  16 ,
		25, 109,  74,   3,    126,  56,   99,   15,   69,   73,   76,   19,   97 ,
		59, 84,   102,  53,   30,   34,   33,   105,  75,   102,  60,   121,  93
	};

	int b[N_mat*N_mat] = {
		102,  61,   111,  79,   99,   3,    25,   50,   33,   48,   5,    94,   28 ,
		106,  89,   35,   37,   112   ,51   ,13,  70,   3,    110,  31,   7,    99 ,
		65, 115,  94,   68,   95,   114,  34,   34,   64,   1,    11,   66,   126 ,
		114,  37,   42,   3,    88,   35,   124   ,50,  74,   95,   25,   34    ,24 ,
		25, 111,  4,    116   ,54,  90    ,11   ,32   ,121, 20,   26,   62,   60 ,
		45, 41,   20,   33,   89,   75,   89,   2,    28    ,19,  96,   46,   119 ,
		39, 68,   87,   59,   33,   82    ,94,  14,   115,  0,    0,    92,   85 ,
		58, 62,   122,  106   ,93,  39,   86,   80,   75,   23,   57,   89,   7 ,
		119,  75,   20,   42,   1,    120,  83,   24,   62,   78,   20,   25,   126 ,
		121,  42,   78,   45,   8,    17,   52,   38,   44,   13,   104,  57,   62 ,
		29, 96,   0,    64,   47,   50,   22,   17,   88,   63,   108,  78,   101 ,
		70, 108,  69,   12,   0,    80,   115   ,107, 71,   54,   5,    57,   3 ,
		123,  72,   56,   5,    30,   45    ,2,   11,   124,  84,   63,   47,   104
	};

	int c[N_mat*N_mat] = { 0 };
	int cslow[N_mat*N_mat] = { 0 };

	int *dev_a, *dev_b, *dev_c, *dev_c_slow;


	//Initialize Timer
	cudaEvent_t start, start1, stop, stop1;
	cudaEventCreate(&start);
	cudaEventCreate(&start1);
	cudaEventCreate(&stop);
	cudaEventCreate(&stop1);

	//Device Info
	cudaDeviceProp prop;
	int device;
	cudaGetDevice(&device);
	cudaGetDeviceProperties(&prop, device);

	printf("  Device name: %s\n", prop.name);
	printf("  clockRate: %i\n", prop.clockRate);
	printf("  warpSize: %i\n", prop.warpSize);
	printf("  multiProcessorCount: %i\n", prop.multiProcessorCount);
	printf("  maxThreadsDim: %ix%ix%i\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
	printf("  maxThreadsPerBlock: %i\n", prop.maxThreadsPerBlock);
	printf("  maxThreadsPerMultiProcessor: %i\n", prop.maxThreadsPerMultiProcessor);
	std::cout << std::endl;




	//Allocating vectors in device memory
	cudaMalloc((void**)&dev_a, N_mat*N_mat * sizeof(int));
	cudaMalloc((void**)&dev_b, N_mat*N_mat * sizeof(int));
	cudaMalloc((void**)&dev_c, N_mat*N_mat * sizeof(int));
	cudaMalloc((void**)&dev_c_slow, N_mat*N_mat * sizeof(int));


	//Copy vectors from host memory to device memory
	cudaMemcpy(dev_a, a, N_mat*N_mat * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, b, N_mat*N_mat * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_c, c, N_mat*N_mat * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_c_slow, c, N_mat*N_mat * sizeof(int), cudaMemcpyHostToDevice);


	//Invoce kernel
	// Setup execution parameters
	dim3 threads(3, N_mat, MAX_acc);
	//dim3 grid(13 / threads.x, 13 / threads.y);

	//Fast Parallel Execution
	cudaEventRecord(start);
	//matrixMulCUDA <<<1,threads>>> (dev_a, dev_b, dev_c);
	matrixMulCUDA << <iterations, threads >> > (dev_a, dev_b, dev_c);
	cudaDeviceSynchronize();
	cudaEventRecord(stop);

	//Occupancy
	int block_size = N_mat*N_mat;
	int output = 1;
	cudaOccupancyMaxActiveBlocksPerMultiprocessor(
		&output,
		matrixMulCUDA,
		block_size,
		0);

	double activeWarps = (double)output * (double)block_size / (double)prop.warpSize;
	double maxWarps = (double)prop.maxThreadsPerMultiProcessor / (double)prop.warpSize;

	std::cout << "Occupancy (fast): " << (double)activeWarps / maxWarps * 100 << "%" << std::endl;
	std::cout << std::endl;

	//Fast print
	std::cout << "Fast Parallel Execution:\n";
	unsigned long mem_size_C = sizeof(int) * N_mat*N_mat;
	cudaMemcpy(c, dev_c, mem_size_C, cudaMemcpyDeviceToHost);
	printmatrix(c);
	printmatrix(a);
	printmatrix(b);


	//Retrieve timer
	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("Time (Fast): %f ms \n", milliseconds);
	std::cout << std::endl;


	//Slow Serial Execution
	if (!DEBUG) {
		block_size = 1;
		cudaEventRecord(start1);
		matrixMulCUDASlow << <iterations, 1 >> > (dev_a, dev_b, dev_c_slow);
		cudaEventRecord(stop1);

		//Occupancy
		int output = 1;
		cudaOccupancyMaxActiveBlocksPerMultiprocessor(
			&output,
			matrixMulCUDASlow,
			block_size,
			0);

		double activeWarps = (double)output * (double)block_size / (double)prop.warpSize;
		double maxWarps = (double)prop.maxThreadsPerMultiProcessor / (double)prop.warpSize;

		std::cout << "Occupancy (slow): " << (double)activeWarps / maxWarps * 100 << "%" << std::endl;
		std::cout << std::endl;


		//Slow Print
		std::cout << "Slow Serial Execution:\n";
		unsigned long mem_size_C = sizeof(int) *N_mat*N_mat;
		cudaMemcpy(cslow, dev_c_slow, mem_size_C, cudaMemcpyDeviceToHost);
		printmatrix(cslow);

		//Retrieve slowtimer
		cudaEventSynchronize(stop1);
		float millisecondsslow = 0;
		cudaEventElapsedTime(&millisecondsslow, start1, stop1);
		printf("Time (Slow): %f ms \n", millisecondsslow);

		std::cout << std::endl;

		printf("Speedup: %f times\n", (millisecondsslow / milliseconds));
	}

	//Free memory
	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);
	cudaFree(dev_c_slow);

	cudaError_t cudaStatus;
	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}


	std::cout << "Press ENTER to exit...";
	std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
	return 0;
}

