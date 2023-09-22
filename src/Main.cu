#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Init.cuh"
#include "OneSweep.cuh"
#include "cub/device/device_radix_sort.cuh"
#include "cub/agent/agent_radix_sort_onesweep.cuh"
#include <stdio.h>

const int sizeExponent = 28;
const int size = 1 << sizeExponent;
const int testIterations = 50;

//Disable this when increasing test iterations, otherwise will be too slow
//because of the device to host readback speed
const int performValidation = true;

const int radix = 256;
const int radixPasses = 4;
const int partitionSize = 7680;
const int globalHistThreadblocks = 2048;
const int binningThreadblocks = size / partitionSize;

const int laneCount = 32;
const int globalHistWarps = 8;
const int digitBinWarps = 16;
dim3 globalHistDim(laneCount, globalHistWarps, 1);
dim3 digitBinDim(laneCount, digitBinWarps, 1);

unsigned int* sort;
unsigned int* alt;
unsigned int* index;
unsigned int* globalHistogram;
unsigned int* firstPassHistogram;
unsigned int* secPassHistogram;
unsigned int* thirdPassHistogram;
unsigned int* fourthPassHistogram;

void InitMemory()
{
	cudaMemset(index, 0, radixPasses * sizeof(unsigned int));
	cudaMemset(globalHistogram, 0, radix * radixPasses * sizeof(unsigned int));
	cudaMemset(firstPassHistogram, 0, radix * binningThreadblocks * sizeof(unsigned int));
	cudaMemset(secPassHistogram, 0, radix * binningThreadblocks * sizeof(unsigned int));
	cudaMemset(thirdPassHistogram, 0, radix * binningThreadblocks * sizeof(unsigned int));
	cudaMemset(fourthPassHistogram, 0, radix * binningThreadblocks * sizeof(unsigned int));
}

void DispatchKernels()
{
	InitMemory();

	k_GlobalHistogram <<<globalHistThreadblocks, globalHistDim>>> (sort, globalHistogram, size);

	k_DigitBinning <<<binningThreadblocks, digitBinDim>>> (globalHistogram, sort, alt,
		firstPassHistogram, index, size, 0);

	k_DigitBinning <<<binningThreadblocks, digitBinDim>>> (globalHistogram, alt, sort,
		secPassHistogram, index, size, 8);

	k_DigitBinning <<<binningThreadblocks, digitBinDim>>> (globalHistogram, sort, alt,
		thirdPassHistogram, index, size, 16);

	k_DigitBinning <<<binningThreadblocks, digitBinDim>>> (globalHistogram, alt, sort,
		fourthPassHistogram, index, size, 24);
}

//Test for correctness
void ValidationTest()
{
	printf("Beginning VALIDATION tests at size %u and %u iterations. \n", size, testIterations);
	unsigned int* validationArray = new unsigned int[size];
	int testsPassed = 0;

	for (int i = 1; i <= testIterations; ++i)
	{
		k_InitRandom <<<256, 1024>>> (sort, size, i);
		DispatchKernels();
		cudaDeviceSynchronize();
		cudaMemcpy(validationArray, sort, size * sizeof(unsigned int), cudaMemcpyDeviceToHost);

		bool isCorrect = true;
		for (int k = 1; k < size; ++k)
		{
			if (validationArray[k] < validationArray[k - 1])
			{
				isCorrect = false;
				break;
			}
		}

		if (isCorrect)
			testsPassed++;
		else
			printf("Test iteration %d failed.", i);
	}

	printf("%d/%d tests passed.\n", testsPassed, testIterations);
	delete[] validationArray;
}

//Discard the first result to prep caches and TLB
void TimingTest()
{
	printf("Beginning TIMING tests at size %u and %u iterations. \n", size, testIterations);
	printf("Running ");

	cudaEvent_t start;
	cudaEvent_t stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	float totalTime = 0.0f;
	for (int i = 0; i <= testIterations; ++i)
	{
		k_InitRandom <<<256, 1024>>> (sort, size, i);
		cudaEventRecord(start);
		DispatchKernels();
		cudaEventRecord(stop);
		cudaEventSynchronize(stop);

		float millis;
		cudaEventElapsedTime(&millis, start, stop);
		if (i)
			totalTime += millis;

		if ((i & 15) == 0)
			printf(". ");
	}

	printf("\n");
	totalTime /= 1000.0f;
	printf("Total time elapsed: %f\n", totalTime);
	printf("Estimated speed at %u 32-bit elements: %E keys/sec\n", size, size / totalTime * testIterations);
}

int main()
{
	cudaMalloc(&sort, size * sizeof(unsigned int));
	cudaMalloc(&alt, size * sizeof(unsigned int));
	cudaMalloc(&index, radixPasses * sizeof(unsigned int));
	cudaMalloc(&globalHistogram, radix * radixPasses * sizeof(unsigned int));
	cudaMalloc(&firstPassHistogram, binningThreadblocks * radix * sizeof(unsigned int));
	cudaMalloc(&secPassHistogram, binningThreadblocks * radix * sizeof(unsigned int));
	cudaMalloc(&thirdPassHistogram, binningThreadblocks * radix * sizeof(unsigned int));
	cudaMalloc(&fourthPassHistogram, binningThreadblocks * radix * sizeof(unsigned int));

	if (performValidation)
		ValidationTest();
	TimingTest();

	cudaFree(sort);
	cudaFree(alt);
	cudaFree(index);
	cudaFree(globalHistogram);
	cudaFree(firstPassHistogram);
	cudaFree(secPassHistogram);
	cudaFree(thirdPassHistogram);
	cudaFree(fourthPassHistogram);
}