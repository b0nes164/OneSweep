# OneSweep
A simple library-less CUDA implementation of the Adinets and Merrill's [OneSweep](https://arxiv.org/abs/2206.01784) sorting algorithm. Given 2^28 uniform random 32-bit keys, our implementation achieves a performance of $\sim$ 8.48 G keys/sec on a 2080 super, for an effective memory bandwidth utilization of 61.5%. This is inline with results from the original paper, which achieves a performance of $\sim$ 29.4 G keys/sec on an A100 80GB for an effective memory bandwidth utilization 54.7%.

The purpose of this repo is to demystify the implmentation of the algorithm. It is not intended for production or use, instead a proper implementation can be found at the [CUB](https://github.com/NVIDIA/cub) library. Notably our implementation lacks: short circuit evaluation, support for data types besides `unsigned int`, support for aligned scattering, and tuning for cards other than the 2080 super.

## Strongly Suggested Reading / Bibliography 
Andy Adinets and Duane Merrill. Onesweep: A Faster Least Significant Digit Radix Sort for GPUs. 2022. arXiv: 2206.01784 [cs.DC]

Duane Merrill and Michael Garland. “Single-pass Parallel Prefix Scan with De-coupled Lookback”. In: 2016. url: https://research.nvidia.com/publication/2016-03_single-pass-parallel-prefix-scan-decoupled-look-back

Saman Ashkiani et al. “GPU Multisplit”. In: SIGPLAN Not. 51.8 (Feb. 2016). issn: 0362-1340. doi: 10.1145/3016078.2851169. url: https://doi.org/10.1145/3016078.2851169.
