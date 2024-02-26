#pragma once
#include <stdio.h>
#include <stdint.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Utils.cuh"

__global__ void GlobalHistogram(
    uint32_t* sort,
    uint32_t* globalHistogram,
    uint32_t size);

__global__ void Scan(
    uint32_t* globalHistogram,
    uint32_t* firstPassHistogram,
    uint32_t* secPassHistogram,
    uint32_t* thirdPassHistogram,
    uint32_t* fourthPassHistogram);

__global__ void DigitBinningPass(
    uint32_t* sort,
    uint32_t* alt,
    volatile uint32_t* passHistogram,
    volatile uint32_t* index,
    uint32_t size,
    uint32_t radixShift);