#pragma once

__global__ void k_InitDescending(unsigned int* sort, int size);

__global__ void k_InitRandom(unsigned int* sort, int size, int seed);