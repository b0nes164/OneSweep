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

#include "OneSweep.cuh"

#define RADIX               256     //Number of digit bins
#define RADIX_MASK          255     //Mask of digit bins, to extract digits
#define RADIX_LOG           8       //log2(RADIX)

#define SEC_RADIX_START     256     //Offset for retrieving value from global histogram buffer
#define THIRD_RADIX_START   512     //Offset for retrieving value from global histogram buffer
#define FOURTH_RADIX_START  768     //Offset for retrieving value from global histogram buffer

//For the upfront global histogram kernel
#define G_HIST_PART_SIZE	65536
#define G_HIST_VEC_SIZE		16384

//For the digit binning
#define BIN_PART_SIZE       7680                                    //Partition tile size in k_DigitBinning
#define BIN_HISTS_SIZE      4096                                    //Total size of warp histograms in shared memory in k_DigitBinning
#define BIN_SUB_PART_SIZE   480                                     //Subpartition tile size of a single warp in k_DigitBinning
#define BIN_WARPS           16                                      //Warps per threadblock in k_DigitBinning
#define BIN_KEYS_PER_THREAD 15                                      //Keys per thread in k_DigitBinning
#define BIN_SUB_PART_START  (WARP_INDEX * BIN_SUB_PART_SIZE)        //Starting offset of a subpartition tile
#define BIN_PART_START      (partitionIndex * BIN_PART_SIZE)        //Starting offset of a partition tile

//for the chained scan with decoupled lookback
#define FLAG_NOT_READY      0                                       //Flag value inidicating neither inclusive sum, nor reduction of a partition tile is ready
#define FLAG_REDUCTION      1                                       //Flag value indicating reduction of a partition tile is ready
#define FLAG_INCLUSIVE      2                                       //Flag value indicating inclusive sum of a partition tile is ready
#define FLAG_MASK           3                                       //Mask used to retrieve flag values

__global__ void k_GlobalHistogram(uint32_t* sort, uint32_t* globalHistogram, uint32_t size)
{
	__shared__ uint32_t s_globalHistFirst[RADIX * 2];
	__shared__ uint32_t s_globalHistSec[RADIX * 2];
	__shared__ uint32_t s_globalHistThird[RADIX * 2];
	__shared__ uint32_t s_globalHistFourth[RADIX * 2];

	//clear shared memory
	for (uint32_t i = threadIdx.x; i < RADIX * 2; i += blockDim.x)
	{
		s_globalHistFirst[i] = 0;
		s_globalHistSec[i] = 0;
		s_globalHistThird[i] = 0;
		s_globalHistFourth[i] = 0;
	}
	__syncthreads();
	
	//histogram
	{
		//64 threads : 1 histogram in shared memory
		uint32_t* s_wavesHistFirst = &s_globalHistFirst[threadIdx.x / 64 * RADIX];
		uint32_t* s_wavesHistSec = &s_globalHistSec[threadIdx.x / 64 * RADIX];
		uint32_t* s_wavesHistThird = &s_globalHistThird[threadIdx.x / 64 * RADIX];
		uint32_t* s_wavesHistFourth = &s_globalHistFourth[threadIdx.x / 64 * RADIX];

		if (blockIdx.x < gridDim.x - 1)
		{
			const uint32_t partEnd = (blockIdx.x + 1) * G_HIST_VEC_SIZE;
			for (uint32_t i = threadIdx.x + (blockIdx.x * G_HIST_VEC_SIZE); i < partEnd; i += blockDim.x)
			{
				uint4 t[1] = { reinterpret_cast<uint4*>(sort)[i] };

				atomicAdd(&s_wavesHistFirst[reinterpret_cast<uint8_t*>(t)[0]], 1);
				atomicAdd(&s_wavesHistSec[reinterpret_cast<uint8_t*>(t)[1]], 1);
				atomicAdd(&s_wavesHistThird[reinterpret_cast<uint8_t*>(t)[2]], 1);
				atomicAdd(&s_wavesHistFourth[reinterpret_cast<uint8_t*>(t)[3]], 1);

				atomicAdd(&s_wavesHistFirst[reinterpret_cast<uint8_t*>(t)[4]], 1);
				atomicAdd(&s_wavesHistSec[reinterpret_cast<uint8_t*>(t)[5]], 1);
				atomicAdd(&s_wavesHistThird[reinterpret_cast<uint8_t*>(t)[6]], 1);
				atomicAdd(&s_wavesHistFourth[reinterpret_cast<uint8_t*>(t)[7]], 1);

				atomicAdd(&s_wavesHistFirst[reinterpret_cast<uint8_t*>(t)[8]], 1);
				atomicAdd(&s_wavesHistSec[reinterpret_cast<uint8_t*>(t)[9]], 1);
				atomicAdd(&s_globalHistThird[reinterpret_cast<uint8_t*>(t)[10]], 1);
				atomicAdd(&s_wavesHistFourth[reinterpret_cast<uint8_t*>(t)[11]], 1);

				atomicAdd(&s_wavesHistFirst[reinterpret_cast<uint8_t*>(t)[12]], 1);
				atomicAdd(&s_wavesHistSec[reinterpret_cast<uint8_t*>(t)[13]], 1);
				atomicAdd(&s_wavesHistThird[reinterpret_cast<uint8_t*>(t)[14]], 1);
				atomicAdd(&s_wavesHistFourth[reinterpret_cast<uint8_t*>(t)[15]], 1);
			}
		}

		if (blockIdx.x == gridDim.x - 1)
		{
			for (uint32_t i = threadIdx.x + (blockIdx.x * G_HIST_PART_SIZE); i < size; i += blockDim.x)
			{
				uint32_t t[1] = { sort[i] };
				atomicAdd(&s_wavesHistFirst[reinterpret_cast<uint8_t*>(t)[0]], 1);
				atomicAdd(&s_wavesHistSec[reinterpret_cast<uint8_t*>(t)[1]], 1);
				atomicAdd(&s_wavesHistThird[reinterpret_cast<uint8_t*>(t)[2]], 1);
				atomicAdd(&s_wavesHistFourth[reinterpret_cast<uint8_t*>(t)[3]], 1);
			}
		}
	}
	__syncthreads();

	//reduce to the first hist
	for (uint32_t i = threadIdx.x; i < RADIX; i += blockDim.x)
	{
		s_globalHistFirst[i] += s_globalHistFirst[i + RADIX];
		s_globalHistSec[i] += s_globalHistSec[i + RADIX];
		s_globalHistThird[i] += s_globalHistThird[i + RADIX];
		s_globalHistFourth[i] += s_globalHistFourth[i + RADIX];
	}

	//exclusive prefix sum over the counts
	for (uint32_t i = threadIdx.x; i < RADIX; i += blockDim.x)
	{
		s_globalHistFirst[i] = InclusiveWarpScanCircularShift(s_globalHistFirst[i]);
		s_globalHistSec[i] = InclusiveWarpScanCircularShift(s_globalHistSec[i]);
		s_globalHistThird[i] = InclusiveWarpScanCircularShift(s_globalHistThird[i]);
		s_globalHistFourth[i] = InclusiveWarpScanCircularShift(s_globalHistFourth[i]);
	}
	__syncthreads();

	if (threadIdx.x < (RADIX >> LANE_LOG))
	{
		s_globalHistFirst[threadIdx.x << LANE_LOG] = ActiveExclusiveWarpScan(s_globalHistFirst[threadIdx.x << LANE_LOG]);
		s_globalHistSec[threadIdx.x << LANE_LOG] = ActiveExclusiveWarpScan(s_globalHistSec[threadIdx.x << LANE_LOG]);
		s_globalHistThird[threadIdx.x << LANE_LOG] = ActiveExclusiveWarpScan(s_globalHistThird[threadIdx.x << LANE_LOG]);
		s_globalHistFourth[threadIdx.x << LANE_LOG] = ActiveExclusiveWarpScan(s_globalHistFourth[threadIdx.x << LANE_LOG]);
	}
	__syncthreads();
	
	//Atomically add to device memory
	for (uint32_t i = threadIdx.x; i < RADIX; i += blockDim.x)
	{
		atomicAdd(&globalHistogram[i], s_globalHistFirst[i] + (getLaneId() ? __shfl_sync(0xfffffffe, s_globalHistFirst[i - 1], 1) : 0));
		atomicAdd(&globalHistogram[i + SEC_RADIX_START], s_globalHistSec[i] + (getLaneId() ? __shfl_sync(0xfffffffe, s_globalHistSec[i - 1], 1) : 0));
		atomicAdd(&globalHistogram[i + THIRD_RADIX_START], s_globalHistThird[i] + (getLaneId() ? __shfl_sync(0xfffffffe, s_globalHistThird[i - 1], 1) : 0));
		atomicAdd(&globalHistogram[i + FOURTH_RADIX_START], s_globalHistFourth[i] + (getLaneId() ? __shfl_sync(0xfffffffe, s_globalHistFourth[i - 1], 1) : 0));
	}
}

__global__ void k_DigitBinning(uint32_t* globalHistogram, uint32_t* sort, uint32_t* alt,
	volatile uint32_t* passHistogram, uint32_t* index, uint32_t size, uint32_t radixShift)
{
	__shared__ uint32_t s_warpHistograms[BIN_PART_SIZE];
	__shared__ uint32_t s_localHistogram[RADIX];
	volatile uint32_t* s_warpHist = &s_warpHistograms[WARP_INDEX << RADIX_LOG];

	//clear shared memory
	for (uint32_t i = threadIdx.x; i < BIN_HISTS_SIZE; i += blockDim.x)  //unnecessary work for last partion but still a win to avoid another barrier
		s_warpHistograms[i] = 0;

	//atomically assign partition tiles
	if (threadIdx.x == 0)
		s_warpHistograms[BIN_PART_SIZE - 1] = atomicAdd(&index[radixShift >> 3], 1);
	__syncthreads();
	const uint32_t partitionIndex = s_warpHistograms[BIN_PART_SIZE - 1];

	//load global histogram into shared memory
	if(threadIdx.x < RADIX)
		s_localHistogram[threadIdx.x] = globalHistogram[threadIdx.x + (radixShift << 5)];

	//To handle input sizes not perfect multiples of the partition tile size
	if (partitionIndex < gridDim.x - 1)
	{
		//load keys
		uint32_t keys[BIN_KEYS_PER_THREAD];
		#pragma unroll
		for (uint32_t i = 0, t = getLaneId() + BIN_SUB_PART_START + BIN_PART_START; i < BIN_KEYS_PER_THREAD; ++i, t += LANE_COUNT)
			keys[i] = sort[t];

		uint32_t _offsets[(BIN_KEYS_PER_THREAD >> 1) + (BIN_KEYS_PER_THREAD & 1 ? 1 : 0)];
		uint16_t* offsets = reinterpret_cast<uint16_t*>(_offsets);

		//WLMS
		#pragma unroll
		for (uint32_t i = 0; i < BIN_KEYS_PER_THREAD; ++i)
		{
			unsigned warpFlags = 0xffffffff;
			for (int k = radixShift; k < radixShift + RADIX_LOG; ++k)
			{
				const bool t2 = keys[i] >> k & 1;
				warpFlags &= (t2 ? 0 : 0xffffffff) ^ __ballot_sync(0xffffffff, t2);
			}

			const uint32_t bits = __popc(warpFlags & getLaneMaskLt());

			uint32_t preIncrementVal;
			if(bits == 0)
				preIncrementVal = atomicAdd((uint32_t*)&s_warpHist[keys[i] >> radixShift & RADIX_MASK], __popc(warpFlags));

			offsets[i] = __shfl_sync(0xffffffff, preIncrementVal, __ffs(warpFlags) - 1) + bits;

			//CUB version
			/*
			offsets[i] = s_warpHist[keys[i] >> radixShift & RADIX_MASK] + bits;
			__syncwarp(0xffffffff);
			if (bits == 0)
				s_warpHist[keys[i] >> radixShift & RADIX_MASK] += __popc(warpFlags);
			__syncwarp(0xffffffff);
			*/
		}
		__syncthreads();

		//exclusive prefix sum up the warp histograms
		if (threadIdx.x < RADIX)
		{
			uint32_t reduction = s_warpHistograms[threadIdx.x];
			for (uint32_t i = threadIdx.x + RADIX; i < BIN_HISTS_SIZE; i += RADIX)
			{
				reduction += s_warpHistograms[i];
				s_warpHistograms[i] = reduction - s_warpHistograms[i];
			}

			atomicAdd((uint32_t*)&passHistogram[threadIdx.x * gridDim.x + partitionIndex], (partitionIndex ? FLAG_REDUCTION : FLAG_INCLUSIVE) | reduction << 2);

			//begin the exclusive prefix sum across the reductions
			s_warpHistograms[threadIdx.x] = InclusiveWarpScanCircularShift(reduction);
		}
		__syncthreads();

		if (threadIdx.x < (RADIX >> LANE_LOG))
			s_warpHistograms[threadIdx.x << LANE_LOG] = ActiveExclusiveWarpScan(s_warpHistograms[threadIdx.x << LANE_LOG]);
		__syncthreads();

		if (threadIdx.x < RADIX && getLaneId())
			s_warpHistograms[threadIdx.x] += __shfl_sync(0xfffffffe, s_warpHistograms[threadIdx.x - 1], 1);
		__syncthreads();

		//update offsets
		if (WARP_INDEX)
		{
			#pragma unroll 
			for (uint32_t i = 0; i < BIN_KEYS_PER_THREAD; ++i)
			{
				const unsigned int t2 = keys[i] >> radixShift & RADIX_MASK;
				offsets[i] += s_warpHist[t2] + s_warpHistograms[t2];
			}
		}
		else
		{
			#pragma unroll
			for (uint32_t i = 0; i < BIN_KEYS_PER_THREAD; ++i)
				offsets[i] += s_warpHistograms[keys[i] >> radixShift & RADIX_MASK];
		}

		//split the warps into single thread cooperative groups and lookback
		if (partitionIndex)
		{
			for (uint32_t i = threadIdx.x; i < RADIX; i += blockDim.x)
			{
				uint32_t reduction = 0;
				for (uint32_t k = partitionIndex; k > 0; )
				{
					const uint32_t flagPayload = passHistogram[i * gridDim.x + k - 1];

					if ((flagPayload & FLAG_MASK) == FLAG_INCLUSIVE)
					{
						reduction += flagPayload >> 2;
						atomicAdd((uint32_t*)&passHistogram[i * gridDim.x + partitionIndex], 1 | (reduction << 2));
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
			if (threadIdx.x < RADIX)
				s_localHistogram[threadIdx.x] -= s_warpHistograms[threadIdx.x];
		}
		__syncthreads();

		//scatter keys into shared memory
		#pragma unroll
		for (uint32_t i = 0; i < BIN_KEYS_PER_THREAD; ++i)
			s_warpHistograms[offsets[i]] = keys[i];
		__syncthreads();

		//scatter runs of keys into device memory
		for (uint32_t i = threadIdx.x; i < BIN_PART_SIZE; i += blockDim.x)
			alt[s_localHistogram[s_warpHistograms[i] >> radixShift & RADIX_MASK] + i] = s_warpHistograms[i];
	}
	
	//Process the final partition slightly differently
	if(partitionIndex == gridDim.x - 1)
	{
		__syncthreads();

		//immediately begin lookback
		if (partitionIndex)
		{
			for (uint32_t i = threadIdx.x; i < RADIX; i += blockDim.x)
			{
				uint32_t reduction = 0;
				for (uint32_t k = partitionIndex; k > 0; )
				{
					const uint32_t flagPayload = passHistogram[i * gridDim.x + k - 1];

					if ((flagPayload & FLAG_MASK) == FLAG_INCLUSIVE)
					{
						reduction += flagPayload >> 2;
						s_localHistogram[i] += reduction;
						break;
					}

					if ((flagPayload & FLAG_MASK) == FLAG_REDUCTION)
					{
						reduction += flagPayload >> 2;
						k--;
					}
				}
			}
			__syncthreads();
		}

		const uint32_t partEnd = BIN_PART_START + BIN_PART_SIZE;
		for (uint32_t i = threadIdx.x + BIN_PART_START; i < partEnd; i += blockDim.x)
		{
			uint32_t key;
			uint32_t offset;
			unsigned warpFlags = 0xffffffff;

			if(i < size)
				key = sort[i];

			for (uint32_t k = radixShift; k < radixShift + RADIX_LOG; ++k)
			{
				const bool t = key >> k & 1;
				warpFlags &= (t ? 0 : 0xffffffff) ^ __ballot_sync(0xffffffff, t);
			}
			const uint32_t bits = __popc(warpFlags & getLaneMaskLt());

			#pragma unroll
			for (uint32_t k = 0; k < BIN_WARPS; ++k)
			{
				uint32_t preIncrementVal;
				if (WARP_INDEX == k && bits == 0 && i < size)
					preIncrementVal = atomicAdd(&s_localHistogram[key >> radixShift & RADIX_MASK], __popc(warpFlags));

				if (WARP_INDEX == k)
					offset = __shfl_sync(0xffffffff, preIncrementVal, __ffs(warpFlags) - 1) + bits;
				__syncthreads();
			}

			if(i < size)
				alt[offset] = key;
		}
	}
}

__global__ void k_Print(unsigned int* toPrint, int size)
{
	for (int i = 0; i < size; ++i)
		printf("%d: %u \n", i, toPrint[i] & 255);
}