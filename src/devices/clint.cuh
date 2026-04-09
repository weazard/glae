#pragma once
#include "../core/hart.cuh"

// Per-hart CLINT state
// Memory map: msip[i] at 0x0 + 4*i, mtimecmp[i] at 0x4000 + 8*i, mtime at 0xBFF8
struct Clint {
    uint32_t msip[MAX_HARTS];
    uint64_t mtimecmp[MAX_HARTS];
};

__device__ uint64_t clint_read(HartState* hart, Clint* c, uint64_t offset, int size) {
    // msip region: offset 0x0000 - 0x03FF
    if (offset < 0x4000) {
        uint32_t hartid = offset / 4;
        if (hartid < MAX_HARTS) return c->msip[hartid] & 1;
        return 0;
    }
    // mtimecmp region: offset 0x4000 - 0xBFF7
    if (offset >= 0x4000 && offset < 0xBFF8) {
        uint32_t hartid = (offset - 0x4000) / 8;
        if (hartid < MAX_HARTS) {
            int byte_off = (offset - 0x4000) % 8;
            if (size == 4) return (c->mtimecmp[hartid] >> (byte_off * 8)) & 0xFFFFFFFF;
            return c->mtimecmp[hartid];
        }
        return 0;
    }
    // mtime: offset 0xBFF8
    if (offset >= 0xBFF8 && offset < 0xC000) {
        uint64_t t = hart->get_mtime();
        int byte_off = (offset - 0xBFF8);
        if (size == 4) return (t >> (byte_off * 8)) & 0xFFFFFFFF;
        return t;
    }
    return 0;
}

__device__ void clint_write(HartState* hart, HartState* all_harts, Clint* c,
                            uint64_t offset, uint64_t val, int size) {
    // msip: write to another hart's software interrupt
    if (offset < 0x4000) {
        uint32_t target = offset / 4;
        if (target < MAX_HARTS) {
            c->msip[target] = val & 1;
            if (val & 1)
                all_harts[target].mip |= MIP_MSIP;
            else
                all_harts[target].mip &= ~MIP_MSIP;
        }
        return;
    }
    // mtimecmp
    if (offset >= 0x4000 && offset < 0xBFF8) {
        uint32_t hartid = (offset - 0x4000) / 8;
        if (hartid < MAX_HARTS) {
            if (size == 4) {
                int shift = ((offset - 0x4000) % 8) * 8;
                uint64_t mask = 0xFFFFFFFFULL << shift;
                c->mtimecmp[hartid] = (c->mtimecmp[hartid] & ~mask) | ((val & 0xFFFFFFFF) << shift);
            } else {
                c->mtimecmp[hartid] = val;
            }
            // Clear timer pending if new compare > current time
            if (hart->get_mtime() < c->mtimecmp[hartid])
                all_harts[hartid].mip &= ~MIP_MTIP;
        }
        return;
    }
}

// Called periodically per-hart to check timer
__device__ void clint_tick(HartState* hart, Clint* c) {
    uint32_t id = (uint32_t)hart->mhartid;
    if (id < MAX_HARTS && hart->get_mtime() >= c->mtimecmp[id])
        hart->mip |= MIP_MTIP;
}
