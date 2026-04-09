#pragma once
#include <cstdint>

// ============================================================
// Per-hart instruction cache
// Each hart (GPU block) gets its own icache slice.
// ============================================================
#define ICACHE_ENTRIES 2048
#define ICACHE_MASK (ICACHE_ENTRIES - 1)

struct ICacheEntry {
    uint64_t pc;
    uint32_t insn;  // decompressed 32-bit instruction
    uint8_t  len;   // original length (2 or 4)
    uint8_t  valid;
};

// Per-hart icache array in global memory (allocated dynamically)
// Indexed as: g_icache_pool[hart_id * ICACHE_ENTRIES + index]
static __device__ ICacheEntry* g_icache_pool = nullptr;
static __device__ int g_num_harts = 0;

__device__ static ICacheEntry* hart_icache() {
    return g_icache_pool + blockIdx.x * ICACHE_ENTRIES;
}

__device__ static void icache_flush() {
    ICacheEntry* ic = hart_icache();
    for (int i = 0; i < ICACHE_ENTRIES; i++)
        ic[i].valid = 0;
}

// Flush ALL harts' icaches (called from host via small kernel)
__global__ static void icache_flush_all_kernel(ICacheEntry* pool, int num_harts) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = num_harts * ICACHE_ENTRIES;
    for (int i = tid; i < total; i += gridDim.x * blockDim.x)
        pool[i].valid = 0;
}
