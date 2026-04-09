#pragma once
#include "../core/hart.cuh"

#define PLIC_NUM_SRC   64
#define PLIC_NUM_CTX   (MAX_HARTS * 2)  // 2 contexts per hart (M-mode, S-mode)

struct Plic {
    uint32_t priority[PLIC_NUM_SRC + 1];
    uint64_t pending;
    uint64_t enable[PLIC_NUM_CTX];
    uint32_t threshold[PLIC_NUM_CTX];
    uint32_t claim[PLIC_NUM_CTX];
};

// For a given hart, S-mode context = hart*2+1, M-mode context = hart*2

__device__ uint32_t plic_highest_pending(Plic* p, int ctx) {
    uint64_t candidates = p->pending & p->enable[ctx];
    uint32_t best_irq = 0;
    uint32_t best_prio = 0;
    while (candidates) {
        int irq = __ffsll(candidates);  // 1-based
        if (irq > 0 && p->priority[irq] > best_prio && p->priority[irq] > p->threshold[ctx]) {
            best_prio = p->priority[irq];
            best_irq = irq;
        }
        candidates &= candidates - 1;  // clear lowest set bit
    }
    return best_irq;
}

__device__ uint64_t plic_read(Plic* p, uint64_t offset) {
    // Priority registers: 0x000000 - 0x000FFF (4 bytes each, source 0..1023)
    if (offset < 0x1000) {
        uint32_t src = offset / 4;
        if (src <= PLIC_NUM_SRC) return p->priority[src];
        return 0;
    }
    // Pending: 0x001000 - 0x001007
    if (offset >= 0x1000 && offset < 0x1008) {
        if (offset == 0x1000) return (uint32_t)(p->pending);
        if (offset == 0x1004) return (uint32_t)(p->pending >> 32);
        return 0;
    }
    // Enable: context N at 0x002000 + N*0x80
    if (offset >= 0x2000 && offset < 0x2000 + PLIC_NUM_CTX * 0x80) {
        int ctx = (offset - 0x2000) / 0x80;
        int reg = (offset - 0x2000 - ctx * 0x80) / 4;
        if (ctx < PLIC_NUM_CTX) {
            if (reg == 0) return (uint32_t)(p->enable[ctx]);
            if (reg == 1) return (uint32_t)(p->enable[ctx] >> 32);
        }
        return 0;
    }
    // Threshold + claim/complete: context N at 0x200000 + N*0x1000
    if (offset >= 0x200000) {
        int ctx = (offset - 0x200000) / 0x1000;
        int reg = (offset - 0x200000 - ctx * 0x1000);
        if (ctx < PLIC_NUM_CTX) {
            if (reg == 0) return p->threshold[ctx];
            if (reg == 4) {
                // Claim: return highest priority pending, clear pending
                uint32_t irq = plic_highest_pending(p, ctx);
                if (irq) p->pending &= ~(1ULL << irq);
                p->claim[ctx] = irq;
                return irq;
            }
        }
    }
    return 0;
}

__device__ void plic_write(Plic* p, uint64_t offset, uint64_t val) {
    if (offset < 0x1000) {
        uint32_t src = offset / 4;
        if (src > 0 && src <= PLIC_NUM_SRC)
            p->priority[src] = (uint32_t)val;
        return;
    }
    if (offset >= 0x2000 && offset < 0x2000 + PLIC_NUM_CTX * 0x80) {
        int ctx = (offset - 0x2000) / 0x80;
        int reg = (offset - 0x2000 - ctx * 0x80) / 4;
        if (ctx < PLIC_NUM_CTX) {
            if (reg == 0) p->enable[ctx] = (p->enable[ctx] & 0xFFFFFFFF00000000ULL) | (uint32_t)val;
            if (reg == 1) p->enable[ctx] = (p->enable[ctx] & 0xFFFFFFFFULL) | ((uint64_t)(uint32_t)val << 32);
        }
        return;
    }
    if (offset >= 0x200000) {
        int ctx = (offset - 0x200000) / 0x1000;
        int reg = (offset - 0x200000 - ctx * 0x1000);
        if (ctx < PLIC_NUM_CTX) {
            if (reg == 0) p->threshold[ctx] = (uint32_t)val;
            if (reg == 4) {
                // Complete: allow the IRQ to pend again
                p->claim[ctx] = 0;
            }
        }
    }
}

__device__ void plic_set_pending(Plic* p, int irq) {
    if (irq > 0 && irq <= PLIC_NUM_SRC)
        p->pending |= (1ULL << irq);
}

// Update hart external interrupt pending based on PLIC state
__device__ void plic_update_ext(HartState* hart, Plic* p) {
    if (plic_highest_pending(p, 1))  // S-mode context
        hart->mip |= MIP_SEIP;
    else
        hart->mip &= ~MIP_SEIP;
    if (plic_highest_pending(p, 0))  // M-mode context
        hart->mip |= MIP_MEIP;
    else
        hart->mip &= ~MIP_MEIP;
}
