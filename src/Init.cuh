#pragma once

__global__ void k_Init(unsigned int* globalHistogram, unsigned int* firstPassHistogram, unsigned int* secPassHistogram,
	unsigned int* thirdPassHistogram, unsigned int* fourthPassHistogram, unsigned int* index, int size, int radix, int radixPasses,
	int threadblocks);

__global__ void k_InitDescending(unsigned int* sort, int size);

__global__ void k_InitRandom(unsigned int* sort, int size, int seed);