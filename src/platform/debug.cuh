#pragma once
#include <cstdint>
#include <cstdio>

// Debug flag — set from host before kernel launch
__device__ static bool g_debug = false;

#define DPRINTF(...) do { if (g_debug) printf(__VA_ARGS__); } while (0)

// Boot stats collected regardless, printed only in debug mode
struct BootStats {
    uint64_t total_insns;
    uint64_t fetch_faults;
    uint64_t load_faults;
    uint64_t store_faults;
    uint64_t tlb_hits;
    uint64_t tlb_misses;
    uint64_t sbi_calls;
    uint64_t timer_irqs;
    uint64_t ext_irqs;
    float    kernel_time_ms;
};
