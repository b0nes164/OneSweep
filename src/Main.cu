#include <stdio.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Init.cuh"
#include "OneSweep.cuh"

const uint32_t size = (1 << 28);
const uint32_t testIterations = 25;

//Disable this when increasing test iterations, otherwise will be too slow
//because of the device to host readback speed
const uint32_t performValidation = true;

const uint32_t radix = 256;
const uint32_t radixPasses = 4;
const uint32_t partitionSize = 7680;
const uint32_t globalHistPartitionSize = 65536;
const uint32_t globalHistThreads = 128;
const uint32_t binningThreads = 512;			//2080 super seems to really like 512 
const uint32_t binningThreadblocks = (size + partitionSize - 1) / partitionSize;
const uint32_t globalHistThreadblocks = (size + globalHistPartitionSize - 1) / globalHistPartitionSize;

uint32_t* sort;
uint32_t* alt;
uint32_t* index;
uint32_t* globalHistogram;
uint32_t* firstPassHistogram;
uint32_t* secPassHistogram;
uint32_t* thirdPassHistogram;
uint32_t* fourthPassHistogram;

void InitMemory()
{
	cudaMemset(index, 0, radixPasses * sizeof(uint32_t));
	cudaMemset(globalHistogram, 0, radix * radixPasses * sizeof(uint32_t));
	cudaMemset(firstPassHistogram, 0, radix * binningThreadblocks * sizeof(uint32_t));
	cudaMemset(secPassHistogram, 0, radix * binningThreadblocks * sizeof(uint32_t));
	cudaMemset(thirdPassHistogram, 0, radix * binningThreadblocks * sizeof(uint32_t));
	cudaMemset(fourthPassHistogram, 0, radix * binningThreadblocks * sizeof(uint32_t));
}

void DispatchKernels()
{
	InitMemory();

	cudaDeviceSynchronize();

	GlobalHistogram <<<globalHistThreadblocks, globalHistThreads >>> (sort, globalHistogram, size);

	Scan <<<radixPasses, radix >>> (globalHistogram, firstPassHistogram, secPassHistogram,
		thirdPassHistogram, fourthPassHistogram);

	DigitBinningPass <<<binningThreadblocks, binningThreads >>> (sort, alt, firstPassHistogram,
		index, size, 0);

	DigitBinningPass <<<binningThreadblocks, binningThreads >>> (alt, sort, secPassHistogram,
		index, size, 8);

	DigitBinningPass <<<binningThreadblocks, binningThreads >>> (sort, alt, thirdPassHistogram,
		index, size, 16);

	DigitBinningPass <<<binningThreadblocks, binningThreads >>> (alt, sort, fourthPassHistogram,
		index, size, 24);
}

//Test for correctness
void ValidationTest()
{
	printf("Beginning VALIDATION tests at size %u and %u iterations. \n", size, testIterations);
	uint32_t* validationArray = new uint32_t[size];
	int testsPassed = 0;

	for (uint32_t i = 1; i <= testIterations; ++i)
	{
		InitRandom <<<256, 1024>>> (sort, size, i);
		DispatchKernels();
		cudaDeviceSynchronize();
		cudaMemcpy(validationArray, sort, size * sizeof(uint32_t), cudaMemcpyDeviceToHost);

		bool isCorrect = true;
		for (uint32_t k = 1; k < size; ++k)
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
	for (uint32_t i = 0; i <= testIterations; ++i)
	{
		InitRandom <<<256, 1024>>> (sort, size, i);
		cudaDeviceSynchronize();
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
	cudaMalloc(&sort, size * sizeof(uint32_t));
	cudaMalloc(&alt, size * sizeof(uint32_t));
	cudaMalloc(&index, radixPasses * sizeof(uint32_t));
	cudaMalloc(&globalHistogram, radix * radixPasses * sizeof(uint32_t));
	cudaMalloc(&firstPassHistogram, binningThreadblocks * radix * sizeof(uint32_t));
	cudaMalloc(&secPassHistogram, binningThreadblocks * radix * sizeof(uint32_t));
	cudaMalloc(&thirdPassHistogram, binningThreadblocks * radix * sizeof(uint32_t));
	cudaMalloc(&fourthPassHistogram, binningThreadblocks * radix * sizeof(uint32_t));

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