#pragma once
#include <cstdint>

// ============================================================
// Instruction Cache — avoids DRAM fetch for hot instruction loops
// ============================================================
#define ICACHE_SIZE 2048
#define ICACHE_MASK (ICACHE_SIZE - 1)

struct ICacheEntry {
    uint64_t pc;
    uint32_t insn;  // decompressed 32-bit instruction
    uint8_t  len;   // original length (2 or 4)
    uint8_t  valid;
};

// Global instruction cache (persists across kernel launches)
static __device__ ICacheEntry g_icache[ICACHE_SIZE];
static __device__ bool g_icache_initialized = false;

__device__ static void icache_flush() {
    for (int i = 0; i < ICACHE_SIZE; i++)
        g_icache[i].valid = 0;
}
