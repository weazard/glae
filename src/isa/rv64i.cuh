#pragma once
#include "../core/hart.cuh"
#include "../priv/mmu.cuh"
#include "../priv/trap.cuh"

// ============================================================
// RV64I Load instructions
// ============================================================
__device__ bool exec_load(HartState* hart, Machine* m, uint32_t insn) {
    int64_t offset = imm_i(insn);
    uint64_t addr = hart->x[rs1(insn)] + offset;
    uint32_t d = rd(insn);
    uint64_t val = 0;
    uint32_t f = funct3(insn);

    int size;
    switch (f) {
    case 0: size = 1; break; // LB
    case 1: size = 2; break; // LH
    case 2: size = 4; break; // LW
    case 3: size = 8; break; // LD
    case 4: size = 1; break; // LBU
    case 5: size = 2; break; // LHU
    case 6: size = 4; break; // LWU
    default:
        take_trap(hart, EXC_ILLEGAL_INSN, hart->pc, insn);
        return true;
    }

    if (!mem_load(hart, m, addr, size, &val)) return true; // trap taken

    // Sign extension
    switch (f) {
    case 0: val = (int64_t)(int8_t)val; break;
    case 1: val = (int64_t)(int16_t)val; break;
    case 2: val = (int64_t)(int32_t)val; break;
    case 3: break; // LD — no extension
    case 4: break; // LBU — zero extended
    case 5: break; // LHU
    case 6: break; // LWU
    }

    if (d != 0) hart->x[d] = val;
    return false;
}

// ============================================================
// RV64I Store instructions
// ============================================================
__device__ bool exec_store(HartState* hart, Machine* m, uint32_t insn) {
    int64_t offset = imm_s(insn);
    uint64_t addr = hart->x[rs1(insn)] + offset;
    uint64_t val = hart->x[rs2(insn)];
    uint32_t f = funct3(insn);

    int size;
    switch (f) {
    case 0: size = 1; break; // SB
    case 1: size = 2; break; // SH
    case 2: size = 4; break; // SW
    case 3: size = 8; break; // SD
    default:
        take_trap(hart, EXC_ILLEGAL_INSN, hart->pc, insn);
        return true;
    }

    if (!mem_store(hart, m, addr, size, val)) return true;
    return false;
}

// ============================================================
// RV64I OP-IMM (register-immediate)
// ============================================================
__device__ void exec_op_imm(HartState* hart, uint32_t insn) {
    uint32_t d = rd(insn);
    uint64_t src = hart->x[rs1(insn)];
    int64_t imm = imm_i(insn);
    uint32_t f = funct3(insn);
    uint64_t result;

    switch (f) {
    case 0: result = src + imm; break;                          // ADDI
    case 1: result = src << (imm & 0x3F); break;               // SLLI
    case 2: result = ((int64_t)src < imm) ? 1 : 0; break;     // SLTI
    case 3: result = (src < (uint64_t)imm) ? 1 : 0; break;    // SLTIU
    case 4: result = src ^ (uint64_t)imm; break;               // XORI
    case 5: // SRLI / SRAI
        if (funct7(insn) & 0x20)
            result = (uint64_t)((int64_t)src >> (imm & 0x3F)); // SRAI
        else
            result = src >> (imm & 0x3F);                       // SRLI
        break;
    case 6: result = src | (uint64_t)imm; break;               // ORI
    case 7: result = src & (uint64_t)imm; break;               // ANDI
    default: result = 0;
    }

    if (d != 0) hart->x[d] = result;
}

// ============================================================
// RV64I OP-IMM-32 (32-bit register-immediate, RV64 only)
// ============================================================
__device__ void exec_op_imm_32(HartState* hart, uint32_t insn) {
    uint32_t d = rd(insn);
    int32_t src = (int32_t)hart->x[rs1(insn)];
    int32_t imm = (int32_t)imm_i(insn);
    uint32_t f = funct3(insn);
    int32_t result;

    switch (f) {
    case 0: result = src + imm; break;                                    // ADDIW
    case 1: result = src << (imm & 0x1F); break;                         // SLLIW
    case 5:
        if (funct7(insn) & 0x20)
            result = src >> (imm & 0x1F);                                 // SRAIW
        else
            result = (int32_t)((uint32_t)src >> (imm & 0x1F));           // SRLIW
        break;
    default: result = 0;
    }

    if (d != 0) hart->x[d] = (int64_t)result; // sign-extend 32→64
}

// ============================================================
// RV64I OP (register-register) — excludes M extension (funct7=1)
// ============================================================
__device__ void exec_op_base(HartState* hart, uint32_t insn) {
    uint32_t d = rd(insn);
    uint64_t a = hart->x[rs1(insn)];
    uint64_t b = hart->x[rs2(insn)];
    uint32_t f = funct3(insn);
    uint32_t f7 = funct7(insn);
    uint64_t result;

    switch (f) {
    case 0: result = (f7 & 0x20) ? a - b : a + b; break;     // ADD/SUB
    case 1: result = a << (b & 0x3F); break;                   // SLL
    case 2: result = ((int64_t)a < (int64_t)b) ? 1 : 0; break; // SLT
    case 3: result = (a < b) ? 1 : 0; break;                   // SLTU
    case 4: result = a ^ b; break;                              // XOR
    case 5:
        if (f7 & 0x20)
            result = (uint64_t)((int64_t)a >> (b & 0x3F));     // SRA
        else
            result = a >> (b & 0x3F);                           // SRL
        break;
    case 6: result = a | b; break;                              // OR
    case 7: result = a & b; break;                              // AND
    default: result = 0;
    }

    if (d != 0) hart->x[d] = result;
}

// ============================================================
// RV64I OP-32 (32-bit register-register) — excludes M ext
// ============================================================
__device__ void exec_op_32_base(HartState* hart, uint32_t insn) {
    uint32_t d = rd(insn);
    int32_t a = (int32_t)hart->x[rs1(insn)];
    int32_t b = (int32_t)hart->x[rs2(insn)];
    uint32_t f = funct3(insn);
    uint32_t f7 = funct7(insn);
    int32_t result;

    switch (f) {
    case 0: result = (f7 & 0x20) ? a - b : a + b; break;          // ADDW/SUBW
    case 1: result = a << (b & 0x1F); break;                        // SLLW
    case 5:
        if (f7 & 0x20)
            result = a >> (b & 0x1F);                                // SRAW
        else
            result = (int32_t)((uint32_t)a >> (b & 0x1F));          // SRLW
        break;
    default: result = 0;
    }

    if (d != 0) hart->x[d] = (int64_t)result;
}

// ============================================================
// RV64I Branch
// ============================================================
__device__ bool exec_branch(HartState* hart, uint32_t insn) {
    uint64_t a = hart->x[rs1(insn)];
    uint64_t b = hart->x[rs2(insn)];
    int64_t offset = imm_b(insn);
    uint32_t f = funct3(insn);
    bool take = false;

    switch (f) {
    case 0: take = (a == b); break;                                  // BEQ
    case 1: take = (a != b); break;                                  // BNE
    case 4: take = ((int64_t)a < (int64_t)b); break;               // BLT
    case 5: take = ((int64_t)a >= (int64_t)b); break;              // BGE
    case 6: take = (a < b); break;                                   // BLTU
    case 7: take = (a >= b); break;                                  // BGEU
    default:
        take_trap(hart, EXC_ILLEGAL_INSN, hart->pc, insn);
        return true;
    }

    if (take) {
        uint64_t target = hart->pc + offset;
        if (target & 1) {
            take_trap(hart, EXC_INSN_MISALIGNED, hart->pc, target);
            return true;
        }
        hart->pc = target;
        return true;
    }
    return false; // not taken — advance PC normally
}
