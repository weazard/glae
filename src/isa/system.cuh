#pragma once
#include "../core/hart.cuh"
#include "../core/icache.cuh"
#include "../priv/trap.cuh"
#include "../priv/csr.cuh"
#include "../priv/sbi.cuh"
#include "../priv/mmu.cuh"

// ============================================================
// SYSTEM instructions (opcode 0x73)
// ============================================================
// Returns true if PC was written (trap, MRET, SRET, ECALL, etc.)
__device__ bool exec_system(HartState* hart, Machine* m, uint32_t insn, int insn_len) {
    uint32_t f = funct3(insn);

    // CSR instructions (funct3 != 0)
    if (f != 0) return exec_csr(hart, insn);

    // funct3 == 0: ECALL, EBREAK, MRET, SRET, WFI, SFENCE.VMA
    uint32_t f7 = funct7(insn);
    uint32_t s2 = rs2(insn);

    if (insn == 0x00000073) {
        // ECALL
        if (hart->priv == PRV_S) {
            // SBI call — handle at emulator level
            if (handle_sbi(hart, m)) {
                hart->pc += insn_len;
                return true;
            }
        }
        uint64_t cause;
        switch (hart->priv) {
        case PRV_U: cause = EXC_ECALL_U; break;
        case PRV_S: cause = EXC_ECALL_S; break;
        case PRV_M: cause = EXC_ECALL_M; break;
        default:    cause = EXC_ECALL_U; break;
        }
        take_trap(hart, cause, hart->pc, 0);
        return true;
    }

    if (insn == 0x00100073) {
        // EBREAK
        take_trap(hart, EXC_BREAKPOINT, hart->pc, hart->pc);
        return true;
    }

    if (insn == 0x30200073) {
        // MRET
        if (hart->priv < PRV_M) {
            take_trap(hart, EXC_ILLEGAL_INSN, hart->pc, insn);
            return true;
        }
        do_mret(hart);
        return true;
    }

    if (insn == 0x10200073) {
        // SRET
        if (hart->priv < PRV_S) {
            take_trap(hart, EXC_ILLEGAL_INSN, hart->pc, insn);
            return true;
        }
        if (hart->priv == PRV_S && (hart->mstatus & MSTATUS_TSR)) {
            take_trap(hart, EXC_ILLEGAL_INSN, hart->pc, insn);
            return true;
        }
        do_sret(hart);
        return true;
    }

    if (insn == 0x10500073) {
        // WFI
        if (hart->priv == PRV_U) {
            take_trap(hart, EXC_ILLEGAL_INSN, hart->pc, insn);
            return true;
        }
        if (hart->priv == PRV_S && (hart->mstatus & MSTATUS_TW)) {
            take_trap(hart, EXC_ILLEGAL_INSN, hart->pc, insn);
            return true;
        }
        // Check if any interrupt is pending
        uint64_t pending = hart->mip & hart->mie;
        if (pending == 0) {
            hart->wfi = 1;
            hart->yield_reason = YIELD_WFI;
        }
        // WFI advances PC even if we enter wait
        hart->pc += insn_len;
        return true;
    }

    // SFENCE.VMA
    if (f7 == 0x09) {
        if (hart->priv < PRV_S) {
            take_trap(hart, EXC_ILLEGAL_INSN, hart->pc, insn);
            return true;
        }
        if (hart->priv == PRV_S && (hart->mstatus & MSTATUS_TVM)) {
            take_trap(hart, EXC_ILLEGAL_INSN, hart->pc, insn);
            return true;
        }
        // Flush TLB and instruction cache
        uint64_t vaddr = hart->x[rs1(insn)];
        if (rs1(insn) == 0) {
            tlb_flush(hart->itlb);
            tlb_flush(hart->dtlb);
            icache_flush();  // full icache flush
        } else {
            tlb_flush_addr(hart->itlb, vaddr >> 12);
            tlb_flush_addr(hart->dtlb, vaddr >> 12);
            // Targeted icache invalidation
            uint32_t base_idx = ((uint32_t)(vaddr >> 1)) & ICACHE_MASK;
            hart_icache()[base_idx].valid = 0;
        }
        return false; // advance PC normally
    }

    // FENCE.I — flush icache
    if ((insn & 0x0000707F) == 0x0000100F) {
        icache_flush();
        return false;
    }

    // PAUSE (Zihintpause) = FENCE with specific encoding
    if (insn == 0x0100000F) return false;

    take_trap(hart, EXC_ILLEGAL_INSN, hart->pc, insn);
    return true;
}
