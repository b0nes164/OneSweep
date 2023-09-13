#pragma once
__global__ void k_GlobalHistogram(unsigned int* sort, unsigned int* globalHistogram, int size);

__global__ void k_DigitBinning(unsigned int* globalHistogram, unsigned int* sort, unsigned int* alt,
	volatile unsigned int* passHistogram, unsigned int* index, int size, unsigned int radixShift);

__global__ void k_Print(unsigned int* toPrint, int size);