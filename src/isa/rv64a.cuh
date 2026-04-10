#pragma once
#include "../core/hart.cuh"
#include "../priv/mmu.cuh"
#include "../priv/trap.cuh"

// ============================================================
// A Extension — Atomics (LR/SC, AMO)
//
// SMP-safe: uses CUDA atomic intrinsics so operations are
// coherent across GPU blocks (SMs). This is critical because
// Linux uses AMO for spinlocks and inter-hart synchronization.
// ============================================================

// Helper: get a pointer into DRAM for atomic operations.
// Returns nullptr if the address is not in DRAM (MMIO).
__device__ void* dram_ptr(Machine* m, uint64_t paddr, int size) {
    if (paddr >= DRAM_BASE && paddr + size <= DRAM_BASE + m->dram_size)
        return m->dram + (paddr - DRAM_BASE);
    return nullptr;
}

__device__ bool exec_amo(HartState* hart, Machine* m, uint32_t insn) {
    uint32_t d = rd(insn);
    uint32_t f = funct3(insn);
    uint32_t f5 = funct5(insn);
    uint64_t addr = hart->x[rs1(insn)];
    uint64_t src = hart->x[rs2(insn)];

    bool is_word = (f == 2);
    bool is_dword = (f == 3);
    if (!is_word && !is_dword) {
        take_trap(hart, EXC_ILLEGAL_INSN, hart->pc, insn);
        return true;
    }

    int size = is_word ? 4 : 8;

    // Alignment check
    if (addr & (size - 1)) {
        take_trap(hart, EXC_STORE_MISALIGNED, hart->pc, addr);
        return true;
    }

    // Translate virtual → physical
    uint64_t paddr;
    if (!translate(hart, m, addr, ACCESS_WRITE, &paddr)) return true;

    void* ptr = dram_ptr(m, paddr, size);
    if (!ptr) {
        take_trap(hart, EXC_STORE_ACCESS_FAULT, hart->pc, addr);
        return true;
    }

    // ---- LR (Load-Reserved) ----
    if (f5 == 0x02) {
        uint64_t val;
        if (is_word) {
            val = (int64_t)(int32_t)atomicAdd((uint32_t*)ptr, 0u); // atomic read
        } else {
            val = (int64_t)atomicAdd((unsigned long long*)ptr, 0ull);
        }
        hart->reservation_addr = paddr;
        hart->reservation_valid = 1;
        __threadfence(); // acquire semantics
        if (d != 0) hart->x[d] = val;
        return false;
    }

    // ---- SC (Store-Conditional) ----
    if (f5 == 0x03) {
        if (hart->reservation_valid && paddr == hart->reservation_addr) {
            // Attempt atomic store via CAS
            if (is_word) {
                uint32_t expected;
                memcpy(&expected, ptr, 4);
                uint32_t desired = (uint32_t)src;
                uint32_t old = atomicCAS((uint32_t*)ptr, expected, desired);
                if (old == expected) {
                    // Success
                    __threadfence(); // release semantics
                    hart->reservation_valid = 0;
                    if (d != 0) hart->x[d] = 0;
                    return false;
                }
            } else {
                unsigned long long expected;
                memcpy(&expected, ptr, 8);
                unsigned long long desired = (unsigned long long)src;
                unsigned long long old = atomicCAS((unsigned long long*)ptr, expected, desired);
                if (old == expected) {
                    __threadfence();
                    hart->reservation_valid = 0;
                    if (d != 0) hart->x[d] = 0;
                    return false;
                }
            }
        }
        // Failure
        hart->reservation_valid = 0;
        if (d != 0) hart->x[d] = 1;
        return false;
    }

    // ---- AMO (Atomic Read-Modify-Write) ----
    uint32_t aq = (insn >> 26) & 1;
    uint32_t rl = (insn >> 25) & 1;

    // Release fence before AMO
    if (rl) __threadfence();

    uint64_t old_val;

    if (is_word) {
        uint32_t* p = (uint32_t*)ptr;
        uint32_t s = (uint32_t)src;
        uint32_t old;

        switch (f5) {
        case 0x01: old = atomicExch(p, s); break;
        case 0x00: old = atomicAdd(p, s); break;
        case 0x04: old = atomicXor(p, s); break;
        case 0x0C: old = atomicAnd(p, s); break;
        case 0x08: old = atomicOr(p, s); break;
        case 0x10: old = atomicMin((int*)p, (int)s); break;
        case 0x14: old = atomicMax((int*)p, (int)s); break;
        case 0x18: old = atomicMin(p, s); break;
        case 0x1C: old = atomicMax(p, s); break;
        default:
            take_trap(hart, EXC_ILLEGAL_INSN, hart->pc, insn);
            return true;
        }
        old_val = (int64_t)(int32_t)old; // sign-extend
    } else {
        unsigned long long* p = (unsigned long long*)ptr;
        unsigned long long s = (unsigned long long)src;
        unsigned long long old;

        switch (f5) {
        case 0x01: old = atomicExch(p, s); break;
        case 0x00: old = atomicAdd(p, s); break;
        case 0x04: old = atomicXor(p, s); break;
        case 0x0C: old = atomicAnd(p, s); break;
        case 0x08: old = atomicOr(p, s); break;
        case 0x10: old = atomicMin((long long*)p, (long long)s); break;
        case 0x14: old = atomicMax((long long*)p, (long long)s); break;
        case 0x18: old = atomicMin(p, s); break;
        case 0x1C: old = atomicMax(p, s); break;
        default:
            take_trap(hart, EXC_ILLEGAL_INSN, hart->pc, insn);
            return true;
        }
        old_val = old;
    }

    // Acquire fence after AMO
    if (aq) __threadfence();

    if (d != 0) hart->x[d] = old_val;
    return false;
}
