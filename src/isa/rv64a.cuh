#pragma once
#include "../core/hart.cuh"
#include "../priv/mmu.cuh"
#include "../priv/trap.cuh"

// ============================================================
// A Extension — Atomics (LR/SC, AMO)
// ============================================================
__device__ bool exec_amo(HartState* hart, Machine* m, uint32_t insn) {
    uint32_t d = rd(insn);
    uint32_t f = funct3(insn);
    uint32_t f5 = funct5(insn);
    uint64_t addr = hart->x[rs1(insn)];
    uint64_t src = hart->x[rs2(insn)];

    bool is_word = (f == 2);   // .W
    bool is_dword = (f == 3);  // .D
    if (!is_word && !is_dword) {
        take_trap(hart, EXC_ILLEGAL_INSN, hart->pc, insn);
        return true;
    }

    int size = is_word ? 4 : 8;

    // LR
    if (f5 == 0x02) {
        uint64_t val = 0;
        if (!mem_load(hart, m, addr, size, &val)) return true;
        if (is_word) val = (int64_t)(int32_t)(uint32_t)val;

        // Set reservation
        uint64_t paddr;
        if (!translate(hart, m, addr, ACCESS_READ, &paddr)) return true;
        hart->reservation_addr = paddr;
        hart->reservation_valid = 1;

        if (d != 0) hart->x[d] = val;
        return false;
    }

    // SC
    if (f5 == 0x03) {
        if (hart->reservation_valid) {
            uint64_t paddr;
            if (!translate(hart, m, addr, ACCESS_WRITE, &paddr)) return true;
            if (paddr == hart->reservation_addr) {
                // Success — store and return 0
                if (!mem_store(hart, m, addr, size, src)) return true;
                hart->reservation_valid = 0;
                if (d != 0) hart->x[d] = 0;
                return false;
            }
        }
        // Failure — return 1
        hart->reservation_valid = 0;
        if (d != 0) hart->x[d] = 1;
        return false;
    }

    // AMO operations: load, compute, store
    uint64_t old_val = 0;
    if (!mem_load(hart, m, addr, size, &old_val)) return true;
    uint64_t signed_old = is_word ? (int64_t)(int32_t)(uint32_t)old_val : old_val;

    uint64_t new_val;
    switch (f5) {
    case 0x01: new_val = src; break;           // AMOSWAP
    case 0x00: new_val = old_val + src; break; // AMOADD
    case 0x04: new_val = old_val ^ src; break; // AMOXOR
    case 0x0C: new_val = old_val & src; break; // AMOAND
    case 0x08: new_val = old_val | src; break; // AMOOR
    case 0x10: // AMOMIN
        if (is_word)
            new_val = ((int32_t)(uint32_t)old_val < (int32_t)(uint32_t)src) ?
                      old_val : src;
        else
            new_val = ((int64_t)old_val < (int64_t)src) ? old_val : src;
        break;
    case 0x14: // AMOMAX
        if (is_word)
            new_val = ((int32_t)(uint32_t)old_val > (int32_t)(uint32_t)src) ?
                      old_val : src;
        else
            new_val = ((int64_t)old_val > (int64_t)src) ? old_val : src;
        break;
    case 0x18: // AMOMINU
        new_val = (old_val < src) ? old_val : src;
        break;
    case 0x1C: // AMOMAXU
        new_val = (old_val > src) ? old_val : src;
        break;
    default:
        take_trap(hart, EXC_ILLEGAL_INSN, hart->pc, insn);
        return true;
    }

    if (is_word) new_val = (uint32_t)new_val;
    if (!mem_store(hart, m, addr, size, new_val)) return true;

    if (d != 0) hart->x[d] = signed_old;
    return false;
}
