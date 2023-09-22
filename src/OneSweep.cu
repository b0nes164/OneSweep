/******************************************************************************
 * OneSweep
 *
 * Author:  Thomas Smith 9/13/2023
 *
 * Based off of Research by:
 *          Andy Adinets, Nvidia Corporation
 *          Duane Merrill, Nvidia Corporation
 *          https://research.nvidia.com/publication/2022-06_onesweep-faster-least-significant-digit-radix-sort-gpus
 *
 * Copyright (c) 2011, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>

using namespace cooperative_groups;

//General macros
#define LANE_COUNT          32      //Threads in a warp
#define LANE_MASK           31      //Mask of the lane count
#define LANE_LOG            5       //log2(LANE_COUNT)

#define RADIX               256     //Number of digit bins
#define RADIX_MASK          255     //Mask of digit bins, to extract digits
#define RADIX_LOG           8       //log2(RADIX)

#define SEC_RADIX           8       //shift value to retrieve digits from the second place
#define THIRD_RADIX         16      //shift value to retrieve digits from the second place
#define FOURTH_RADIX        24      //shift value to retrieve digits from the second place

#define SEC_RADIX_START     256     //Offset for retrieving value from global histogram buffer
#define THIRD_RADIX_START   512     //Offset for retrieving value from global histogram buffer
#define FOURTH_RADIX_START  768     //Offset for retrieving value from global histogram buffer

#define LANE                threadIdx.x                             //Lane of a thread
#define WARP_INDEX          threadIdx.y                             //Warp of a thread
#define THREAD_ID           (LANE + (WARP_INDEX << LANE_LOG))       //Threadid

//For the upfront global histogram kernel
#define G_HIST_WARPS        8                                       //Warps per threadblock in k_GlobalHistogram
#define G_HIST_THREADS      256                                     //Threads per threadblock in k_GlobalHistogram
#define G_TBLOCK_LOG        11                                      //log2(gridDim.x)
#define G_HIST_PART_SIZE    (size >> G_TBLOCK_LOG)                  //Partition tile size in k_GlobalHistogram
#define G_HIST_PART_START   (blockIdx.x * G_HIST_PART_SIZE)         //Starting offset of a partition tile
#define G_HIST_PART_END     (blockIdx.x == gridDim.x - 1 ? \
                            size : \
                            (blockIdx.x + 1) * G_HIST_PART_SIZE)

//For the digit binning
#define BIN_PART_SIZE       7680                                    //Partition tile size in k_DigitBinning
#define BIN_HISTS_SIZE      4096                                    //Total size of warp histograms in shared memory in k_DigitBinning
#define BIN_SUB_PART_SIZE   480                                     //Subpartition tile size of a single warp in k_DigitBinning
#define BIN_THREADS         512                                     //Threads per threadblock in k_DigitBinning
#define BIN_WARPS           16                                      //Warps per threadblock in k_DigitBinning
#define BIN_KEYS_PER_THREAD 15                                      //Keys per thread in k_DigitBinning
#define BIN_SUB_PART_START  (WARP_INDEX * BIN_SUB_PART_SIZE)        //Starting offset of a subpartition tile
#define BIN_PART_START      (partitionIndex * BIN_PART_SIZE)        //Starting offset of a partition tile

//for the chained scan with decoupled lookback
#define FLAG_NOT_READY      0                                       //Flag value inidicating neither inclusive sum, nor reduction of a partition tile is ready
#define FLAG_REDUCTION      1                                       //Flag value indicating reduction of a partition tile is ready
#define FLAG_INCLUSIVE      2                                       //Flag value indicating inclusive sum of a partition tile is ready
#define FLAG_MASK           3                                       //Mask used to retrieve flag values

__device__ __forceinline__ void InclusiveWarpScan(volatile unsigned int* t, int index, int strideLog)
{
	if (LANE > 0) t[index] += t[index - (1 << strideLog)];
	if (LANE > 1) t[index] += t[index - (2 << strideLog)];
	if (LANE > 3) t[index] += t[index - (4 << strideLog)];
	if (LANE > 7) t[index] += t[index - (8 << strideLog)];
	if (LANE > 15) t[index] += t[index - (16 << strideLog)];
}

__device__ __forceinline__ void InclusiveWarpScanCircularShift(volatile unsigned int* t, int index)
{
	if (LANE > 0) t[index] += t[index - 1];
	if (LANE > 1) t[index] += t[index - 2];
	if (LANE > 3) t[index] += t[index - 4];
	if (LANE > 7) t[index] += t[index - 8];
	if (LANE > 15) t[index] += t[index - 16];

	t[index] = __shfl_sync(__activemask(), t[index], LANE + LANE_MASK & LANE_MASK);
}

__device__ __forceinline__ void ExclusiveWarpScan(volatile unsigned int* t, int index, int strideLog)
{
	if (LANE > 0) t[index] += t[index - (1 << strideLog)];
	if (LANE > 1) t[index] += t[index - (2 << strideLog)];
	if (LANE > 3) t[index] += t[index - (4 << strideLog)];
	if (LANE > 7) t[index] += t[index - (8 << strideLog)];
	if (LANE > 15) t[index] += t[index - (16 << strideLog)];

	t[index] = LANE ? t[index - (1 << strideLog)] : 0;
}

__global__ void k_GlobalHistogram(unsigned int* sort, unsigned int* globalHistogram, int size)
{
	__shared__ unsigned int s_globalHistFirst[RADIX];
	__shared__ unsigned int s_globalHistSec[RADIX];
	__shared__ unsigned int s_globalHistThird[RADIX];
	__shared__ unsigned int s_globalHistFourth[RADIX];

	//clear
	for (int i = THREAD_ID; i < RADIX; i += G_HIST_THREADS)
	{
		s_globalHistFirst[i] = 0;
		s_globalHistSec[i] = 0;
		s_globalHistThird[i] = 0;
		s_globalHistFourth[i] = 0;
	}
	__syncthreads();

	//Histogram
	{
		const int partitionEnd = G_HIST_PART_END;
		for (int i = THREAD_ID + G_HIST_PART_START; i < partitionEnd; i += G_HIST_THREADS)
		{
			const unsigned int key = sort[i];
			atomicAdd(&s_globalHistFirst[key & RADIX_MASK], 1);
			atomicAdd(&s_globalHistSec[key >> SEC_RADIX & RADIX_MASK], 1);
			atomicAdd(&s_globalHistThird[key >> THIRD_RADIX & RADIX_MASK], 1);
			atomicAdd(&s_globalHistFourth[key >> FOURTH_RADIX], 1);
		}
	}
	__syncthreads();

	//exclusive prefix sum over the counts
	for (int i = THREAD_ID; i < RADIX; i += G_HIST_THREADS)
	{
		InclusiveWarpScanCircularShift(s_globalHistFirst, i);
		InclusiveWarpScanCircularShift(s_globalHistSec, i);
		InclusiveWarpScanCircularShift(s_globalHistThird, i);
		InclusiveWarpScanCircularShift(s_globalHistFourth, i);
	}
	__syncthreads();

	if (LANE < (RADIX >> LANE_LOG) && WARP_INDEX == 0)
	{
		InclusiveWarpScan(s_globalHistFirst, (LANE << LANE_LOG), LANE_LOG);
		InclusiveWarpScan(s_globalHistSec, (LANE << LANE_LOG), LANE_LOG);
		InclusiveWarpScan(s_globalHistThird, (LANE << LANE_LOG), LANE_LOG);
		InclusiveWarpScan(s_globalHistFourth, (LANE << LANE_LOG), LANE_LOG);
	}
	__syncthreads();

	//Atomically add to device memory
	{
		int i = THREAD_ID;
		atomicAdd(&globalHistogram[i], (LANE ? s_globalHistFirst[i] : 0) + (WARP_INDEX ? __shfl_sync(0xffffffff, s_globalHistFirst[i - LANE_COUNT], 0) : 0));
		atomicAdd(&globalHistogram[i + SEC_RADIX_START], (LANE ? s_globalHistSec[i] : 0) + (WARP_INDEX ? __shfl_sync(0xffffffff, s_globalHistSec[i - LANE_COUNT], 0) : 0));
		atomicAdd(&globalHistogram[i + THIRD_RADIX_START], (LANE ? s_globalHistThird[i] : 0) + (WARP_INDEX ? __shfl_sync(0xffffffff, s_globalHistThird[i - LANE_COUNT], 0) : 0));
		atomicAdd(&globalHistogram[i + FOURTH_RADIX_START], (LANE ? s_globalHistFourth[i] : 0) + (WARP_INDEX ? __shfl_sync(0xffffffff, s_globalHistFourth[i - LANE_COUNT], 0) : 0));

		for (i += G_HIST_THREADS; i < RADIX; i += G_HIST_THREADS)
		{
			atomicAdd(&globalHistogram[i], (LANE ? s_globalHistFirst[i] : 0) + __shfl_sync(0xffffffff, s_globalHistFirst[i - LANE_COUNT], 0));
			atomicAdd(&globalHistogram[i + SEC_RADIX_START], (LANE ? s_globalHistSec[i] : 0) + __shfl_sync(0xffffffff, s_globalHistSec[i - LANE_COUNT], 0));
			atomicAdd(&globalHistogram[i + THIRD_RADIX_START], (LANE ? s_globalHistThird[i] : 0) + __shfl_sync(0xffffffff, s_globalHistThird[i - LANE_COUNT], 0));
			atomicAdd(&globalHistogram[i + FOURTH_RADIX_START], (LANE ? s_globalHistFourth[i] : 0) + __shfl_sync(0xffffffff, s_globalHistFourth[i - LANE_COUNT], 0));
		}
	}
}

__global__ void k_DigitBinning(unsigned int* globalHistogram, unsigned int* sort, unsigned int* alt,
	volatile unsigned int* passHistogram, unsigned int* index, int size, unsigned int radixShift)
{
	__shared__ unsigned int s_warpHistograms[BIN_PART_SIZE];
	__shared__ unsigned int s_localHistogram[RADIX];
	unsigned int* s_warpHist = &s_warpHistograms[WARP_INDEX << RADIX_LOG];

	//atomically assign partition tiles
	if (LANE == 0 && WARP_INDEX == 0)
		s_localHistogram[0] = atomicAdd(&index[radixShift >> 3], 1);
	__syncthreads();
	int partitionIndex = s_localHistogram[0];
	__syncthreads();

	//load global histogram into shared memory
	if(THREAD_ID < RADIX)
		s_localHistogram[THREAD_ID] = globalHistogram[THREAD_ID + (radixShift << 5)];

	//clear
	#pragma unroll
	for (int i = LANE; i < RADIX; i += LANE_COUNT)
		s_warpHist[i] = 0;

	//load keys
	unsigned int keys[BIN_KEYS_PER_THREAD];
	#pragma unroll
	for (int i = 0, t = LANE + BIN_SUB_PART_START + BIN_PART_START; i < BIN_KEYS_PER_THREAD; ++i, t += LANE_COUNT)
		keys[i] = sort[t];

	//WLMS
	unsigned int _offsets[(BIN_KEYS_PER_THREAD >> 1) + (BIN_KEYS_PER_THREAD & 1 ? 1 : 0)];
	unsigned short* offsets = reinterpret_cast<unsigned short*>(_offsets);
	for (int i = 0; i < BIN_KEYS_PER_THREAD; ++i)
	{
		unsigned int warpFlags = 0xffffffff;
		for (int k = radixShift; k < radixShift + RADIX_LOG; ++k)
		{
			const bool t2 = keys[i] >> k & 1;
			warpFlags &= (t2 ? 0 : 0xffffffff) ^ __ballot_sync(0xffffffff, t2);
		}

		const unsigned int bits = __popc(warpFlags << LANE_MASK - LANE);
		offsets[i] = s_warpHist[keys[i] >> radixShift & RADIX_MASK] + bits - 1;
		if (bits == 1)
			s_warpHist[keys[i] >> radixShift & RADIX_MASK] += __popc(warpFlags);
	}
	__syncthreads();
	
	//exclusive prefix sum across the warp histograms
	if(THREAD_ID < RADIX)
	{
		const unsigned int t = THREAD_ID;
		for (int i = t + RADIX; i < BIN_HISTS_SIZE; i += RADIX)
		{
			s_warpHistograms[t] += s_warpHistograms[i];
			s_warpHistograms[i] = s_warpHistograms[t] - s_warpHistograms[i];
		}

		if (partitionIndex == 0)
			atomicAdd((unsigned int*)&passHistogram[THREAD_ID * gridDim.x + partitionIndex], FLAG_INCLUSIVE | s_warpHistograms[THREAD_ID] << 2);
		else
			atomicAdd((unsigned int*)&passHistogram[THREAD_ID * gridDim.x + partitionIndex], FLAG_REDUCTION | s_warpHistograms[THREAD_ID] << 2);
	}
	__syncthreads();

	//exlusive prefix sum across the reductions
	if (THREAD_ID < RADIX)
		InclusiveWarpScanCircularShift(s_warpHistograms, THREAD_ID);
	__syncthreads();

	if (LANE < (RADIX >> LANE_LOG) && WARP_INDEX == 0)
		ExclusiveWarpScan(s_warpHistograms, LANE << LANE_LOG, LANE_LOG);
	__syncthreads();

	if (THREAD_ID < RADIX && LANE)
		s_warpHistograms[THREAD_ID] += __shfl_sync(0xfffffffe, s_warpHistograms[THREAD_ID - 1], 1);
	__syncthreads();

	//update offsets
	if (WARP_INDEX)
	{
		#pragma unroll 
		for (int i = 0; i < BIN_KEYS_PER_THREAD; ++i)
		{
			const unsigned int t2 = keys[i] >> radixShift & RADIX_MASK;
			offsets[i] += s_warpHist[t2] + s_warpHistograms[t2];
		}
	}
	else
	{
		#pragma unroll
		for (int i = 0; i < BIN_KEYS_PER_THREAD; ++i)
			offsets[i] += s_warpHistograms[keys[i] >> radixShift & RADIX_MASK];
	}
	__syncthreads();

	//split the warps into single thread cooperative groups and lookback
	if (partitionIndex)
	{
		thread_block_tile<1> threadID = tiled_partition<1>(this_thread_block());

		for (int i = threadID.meta_group_rank(); i < RADIX; i += BIN_THREADS)
		{
			unsigned int reduction = 0;
			for (int k = partitionIndex - 1; 0 <= k;)
			{
				const unsigned int flagPayload = passHistogram[i * gridDim.x + k];

				if ((flagPayload & FLAG_MASK) == FLAG_INCLUSIVE)
				{
					reduction += flagPayload >> 2;
					atomicAdd((unsigned int*)&passHistogram[i * gridDim.x + partitionIndex], 1 | (reduction << 2));
					s_localHistogram[i] += reduction - s_warpHistograms[i];
					break;
				}

				if ((flagPayload & FLAG_MASK) == FLAG_REDUCTION)
				{
					reduction += flagPayload >> 2;
					k--;
				}
			}
		}
	}
	else
	{
		if (THREAD_ID < RADIX)
			s_localHistogram[THREAD_ID] -= s_warpHistograms[THREAD_ID];
	}
	__syncthreads();

	//scatter keys into shared memory
	#pragma unroll
	for (int i = 0; i < BIN_KEYS_PER_THREAD; ++i)
		s_warpHistograms[offsets[i]] = keys[i];
	__syncthreads();

	//scatter runs of keys into device memory
	for (int i = THREAD_ID; i < BIN_PART_SIZE; i += BIN_THREADS)
		alt[s_localHistogram[s_warpHistograms[i] >> radixShift & RADIX_MASK] + i] = s_warpHistograms[i];
	
	//To handle input sizes not perfect multiples of the partition tile size
	if (partitionIndex == gridDim.x - 1)
	{
		__syncthreads();
		{
			const int tid = THREAD_ID;
			if (tid < RADIX)
				s_localHistogram[tid] = (passHistogram[tid * gridDim.x + partitionIndex] >> 2) + globalHistogram[tid + (radixShift << 5)];
		}
		__syncthreads();

		partitionIndex++;
		for (int i = THREAD_ID + BIN_PART_START; i < size; i += BIN_THREADS)
		{
			const unsigned int key = sort[i];
			unsigned int offset = 0xffffffff;

			for (int k = radixShift; k < radixShift + RADIX_LOG; ++k)
			{
				const bool t = key >> k & 1;
				offset &= (t ? 0 : 0xffffffff) ^ __ballot_sync(__activemask(), t);
			}

			#pragma unroll
			for (int k = 0; k < BIN_WARPS; ++k)
			{
				if (WARP_INDEX == k)
				{
					const unsigned int t = s_localHistogram[key >> radixShift & RADIX_MASK];
					const unsigned int bits = __popc(offset << LANE_MASK - LANE);
					if (bits == 1)
						s_localHistogram[key >> radixShift & RADIX_MASK] += __popc(offset);
					offset = t + bits - 1;
				}
				__syncthreads();
			}

			alt[offset] = key;
		}
	}
}

__global__ void k_Print(unsigned int* toPrint, int size)
{
	for (int i = 0; i < size; ++i)
		printf("%d: %u \n", i, toPrint[i]);
}