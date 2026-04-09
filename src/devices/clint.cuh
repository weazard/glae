#pragma once
#include "../core/hart.cuh"

struct Clint {
    uint32_t msip;
    uint64_t mtimecmp;
};

__device__ uint64_t clint_read(HartState* hart, Clint* c, uint64_t offset, int size) {
    if (offset < 4) {
        return c->msip & 1;
    } else if (offset >= 0x4000 && offset < 0x4008) {
        // mtimecmp
        int shift = (offset - 0x4000) * 8;
        if (size == 4) return (c->mtimecmp >> shift) & 0xFFFFFFFF;
        return c->mtimecmp;
    } else if (offset >= 0xBFF8 && offset < 0xC000) {
        // mtime
        uint64_t t = hart->get_mtime();
        int shift = (offset - 0xBFF8) * 8;
        if (size == 4) return (t >> shift) & 0xFFFFFFFF;
        return t;
    }
    return 0;
}

__device__ void clint_write(HartState* hart, Clint* c, uint64_t offset, uint64_t val, int size) {
    if (offset < 4) {
        c->msip = val & 1;
        if (val & 1) hart->mip |= MIP_MSIP;
        else         hart->mip &= ~MIP_MSIP;
    } else if (offset >= 0x4000 && offset < 0x4008) {
        if (size == 4) {
            int shift = (offset - 0x4000) * 8;
            uint64_t mask = 0xFFFFFFFFULL << shift;
            c->mtimecmp = (c->mtimecmp & ~mask) | ((val & 0xFFFFFFFF) << shift);
        } else {
            c->mtimecmp = val;
        }
        // Clear timer pending if new compare > current time
        if (hart->get_mtime() < c->mtimecmp)
            hart->mip &= ~MIP_MTIP;
    }
}

// Called periodically to check timer
__device__ void clint_tick(HartState* hart, Clint* c) {
    if (hart->get_mtime() >= c->mtimecmp)
        hart->mip |= MIP_MTIP;
}
