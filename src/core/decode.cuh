#pragma once
#include "../core/hart.cuh"
#include "../devices/bus.cuh"
#include "../isa/rv64i.cuh"
#include "../isa/rv64m.cuh"
#include "../isa/rv64a.cuh"
#include "../isa/rv64f.cuh"
#include "../isa/rv64d.cuh"
#include "../isa/rv64zb.cuh"
#include "../isa/system.cuh"

// ============================================================
// Top-level instruction decode + execute
// Returns true if PC was already updated (branch/jump/trap)
// ============================================================
__device__ bool execute(HartState* hart, Machine* m, uint32_t insn, int insn_len) {
    uint32_t op = opcode(insn);

    switch (op) {
    case OP_LUI: {
        uint32_t d = rd(insn);
        if (d != 0) hart->x[d] = (uint64_t)imm_u(insn);
        return false;
    }
    case OP_AUIPC: {
        uint32_t d = rd(insn);
        if (d != 0) hart->x[d] = hart->pc + (uint64_t)imm_u(insn);
        return false;
    }
    case OP_JAL: {
        uint32_t d = rd(insn);
        int64_t offset = imm_j(insn);
        uint64_t target = hart->pc + offset;
        if (target & 1) {
            take_trap(hart, EXC_INSN_MISALIGNED, hart->pc, target);
            return true;
        }
        if (d != 0) hart->x[d] = hart->pc + insn_len;
        hart->pc = target;
        return true;
    }
    case OP_JALR: {
        uint32_t d = rd(insn);
        int64_t offset = imm_i(insn);
        uint64_t target = (hart->x[rs1(insn)] + offset) & ~1ULL;
        if (target & 1) {
            take_trap(hart, EXC_INSN_MISALIGNED, hart->pc, target);
            return true;
        }
        if (d != 0) hart->x[d] = hart->pc + insn_len;
        hart->pc = target;
        return true;
    }
    case OP_BRANCH:
        return exec_branch(hart, insn);

    case OP_LOAD:
        return exec_load(hart, m, insn);

    case OP_STORE:
        return exec_store(hart, m, insn);

    case OP_OP_IMM:
        // Try Zbb/Zbs immediate forms first (they share this opcode)
        if (try_exec_zbb_imm(hart, insn)) return false;
        if (try_exec_zbs_imm(hart, insn)) return false;
        exec_op_imm(hart, insn);
        return false;

    case OP_OP_IMM_32:
        if (try_exec_zbb_imm32(hart, insn)) return false;
        if (try_exec_zba_imm32(hart, insn)) return false;
        exec_op_imm_32(hart, insn);
        return false;

    case OP_OP: {
        uint32_t f7 = funct7(insn);
        if (f7 == 0x01) { exec_mul(hart, insn); return false; }
        // Try Zba/Zbb/Zbs (they use various funct7 values on OP)
        if (try_exec_zba_op(hart, insn)) return false;
        if (try_exec_zbb_op(hart, insn)) return false;
        if (try_exec_zbs_op(hart, insn)) return false;
        exec_op_base(hart, insn);
        return false;
    }

    case OP_OP_32: {
        uint32_t f7 = funct7(insn);
        if (f7 == 0x01) { exec_mul_32(hart, insn); return false; }
        if (try_exec_zba_op32(hart, insn)) return false;
        if (try_exec_zbb_op32(hart, insn)) return false;
        exec_op_32_base(hart, insn);
        return false;
    }

    case OP_MISC_MEM:
        // FENCE variants + CBO instructions
        if (try_exec_cbo(hart, m, insn)) return false;
        return false;  // plain FENCE = NOP

    case OP_SYSTEM:
        return exec_system(hart, m, insn, insn_len);

    case OP_AMO:
        return exec_amo(hart, m, insn);

    // --- Floating point ---
    case OP_LOAD_FP:
        return exec_load_fp(hart, m, insn);

    case OP_STORE_FP:
        return exec_store_fp(hart, m, insn);

    case OP_OP_FP: {
        uint32_t fmt = (insn >> 25) & 3;
        if (fmt == 0) return exec_op_fp_s(hart, insn);
        if (fmt == 1) return exec_op_fp_d(hart, insn);
        uint32_t f7 = funct7(insn);
        if (f7 == 0x20) return exec_op_fp_s(hart, insn);
        if (f7 == 0x21) return exec_op_fp_d(hart, insn);
        take_trap(hart, EXC_ILLEGAL_INSN, hart->pc, insn);
        return true;
    }

    case OP_FMADD:
    case OP_FMSUB:
    case OP_FNMSUB:
    case OP_FNMADD: {
        uint32_t fmt = (insn >> 25) & 3;
        if (fmt == 0) return exec_fma_s(hart, insn, op);
        if (fmt == 1) return exec_fma_d(hart, insn, op);
        take_trap(hart, EXC_ILLEGAL_INSN, hart->pc, insn);
        return true;
    }

    default:
        take_trap(hart, EXC_ILLEGAL_INSN, hart->pc, insn);
        return true;
    }
}
