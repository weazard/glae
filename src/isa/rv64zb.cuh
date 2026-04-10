#pragma once
#include "../core/hart.cuh"

// ============================================================
// Zba (Address generation), Zbb (Basic bit-manipulation),
// Zbs (Single-bit), Zicboz (Cache-block zero)
//
// These share opcodes with RV64I (OP, OP-IMM, OP-32, OP-IMM-32)
// but use distinct funct7 values.
// ============================================================

// ---- Zba: shift-and-add operations ----

// Returns true if this funct7 + funct3 + opcode is a Zba instruction on OP (0x33)
__device__ bool try_exec_zba_op(HartState* hart, uint32_t insn) {
    uint32_t f7 = funct7(insn), f3 = funct3(insn);
    uint32_t d = rd(insn);
    uint64_t a = hart->x[rs1(insn)], b = hart->x[rs2(insn)];
    uint64_t result;

    if (f7 == 0x10) {
        switch (f3) {
        case 2: result = (a << 1) + b; break;  // SH1ADD
        case 4: result = (a << 2) + b; break;  // SH2ADD
        case 6: result = (a << 3) + b; break;  // SH3ADD
        default: return false;
        }
    } else return false;

    if (d) hart->x[d] = result;
    return true;
}

// Zba on OP-32 (0x3B)
__device__ bool try_exec_zba_op32(HartState* hart, uint32_t insn) {
    uint32_t f7 = funct7(insn), f3 = funct3(insn);
    uint32_t d = rd(insn);
    uint64_t a = hart->x[rs1(insn)], b = hart->x[rs2(insn)];
    uint64_t result;

    if (f7 == 0x04 && f3 == 0) {
        // ADD.UW: rd = rs2 + zext32(rs1)
        result = b + (uint32_t)a;
    } else if (f7 == 0x10) {
        uint32_t uw = (uint32_t)a;
        switch (f3) {
        case 2: result = ((uint64_t)uw << 1) + b; break;  // SH1ADD.UW
        case 4: result = ((uint64_t)uw << 2) + b; break;  // SH2ADD.UW
        case 6: result = ((uint64_t)uw << 3) + b; break;  // SH3ADD.UW
        default: return false;
        }
    } else return false;

    if (d) hart->x[d] = result;
    return true;
}

// Zba SLLI.UW on OP-IMM-32 (0x1B)
__device__ bool try_exec_zba_imm32(HartState* hart, uint32_t insn) {
    uint32_t f7_top = (insn >> 26) & 0x3F;  // bits 31:26
    uint32_t f3 = funct3(insn);
    if (f7_top == 0x02 && f3 == 1) {
        // SLLI.UW: rd = zext32(rs1) << shamt
        uint32_t shamt = (insn >> 20) & 0x3F;
        uint32_t d = rd(insn);
        uint64_t result = (uint64_t)(uint32_t)hart->x[rs1(insn)] << shamt;
        if (d) hart->x[d] = result;
        return true;
    }
    return false;
}

// ---- Zbb: basic bit manipulation ----

__device__ bool try_exec_zbb_op(HartState* hart, uint32_t insn) {
    uint32_t f7 = funct7(insn), f3 = funct3(insn);
    uint32_t d = rd(insn);
    uint64_t a = hart->x[rs1(insn)], b = hart->x[rs2(insn)];
    uint64_t result;

    if (f7 == 0x20) {
        switch (f3) {
        case 7: result = a & ~b; break;                                      // ANDN
        case 6: result = a | ~b; break;                                      // ORN
        case 4: result = ~(a ^ b); break;                                    // XNOR
        default: return false;
        }
    } else if (f7 == 0x05) {
        switch (f3) {
        case 4: result = ((int64_t)a < (int64_t)b) ? a : b; break;         // MIN
        case 5: result = (a < b) ? a : b; break;                            // MINU
        case 6: result = ((int64_t)a > (int64_t)b) ? a : b; break;         // MAX
        case 7: result = (a > b) ? a : b; break;                            // MAXU
        default: return false;
        }
    } else if (f7 == 0x30) {
        switch (f3) {
        case 1: result = (a << (b & 63)) | (a >> ((64 - (b & 63)) & 63)); break; // ROL
        case 5: result = (a >> (b & 63)) | (a << ((64 - (b & 63)) & 63));  // ROR
                break;
        default: return false;
        }
    } else return false;

    if (d) hart->x[d] = result;
    return true;
}

__device__ bool try_exec_zbb_op32(HartState* hart, uint32_t insn) {
    uint32_t f7 = funct7(insn), f3 = funct3(insn);
    uint32_t d = rd(insn);
    uint32_t a = (uint32_t)hart->x[rs1(insn)], b = (uint32_t)hart->x[rs2(insn)];
    int32_t result;

    if (f7 == 0x04 && f3 == 4) {
        // ZEXT.H (pack with rs2=0)
        if (d) hart->x[d] = (uint64_t)(uint16_t)hart->x[rs1(insn)];
        return true;
    }
    if (f7 == 0x30) {
        switch (f3) {
        case 1: result = (int32_t)((a << (b & 31)) | (a >> ((32 - (b & 31)) & 31))); break; // ROLW
        case 5: result = (int32_t)((a >> (b & 31)) | (a << ((32 - (b & 31)) & 31))); break; // RORW
        default: return false;
        }
    } else return false;

    if (d) hart->x[d] = (int64_t)result;
    return true;
}

__device__ bool try_exec_zbb_imm(HartState* hart, uint32_t insn) {
    uint32_t f3 = funct3(insn);
    uint32_t d = rd(insn);
    uint64_t a = hart->x[rs1(insn)];
    uint64_t result;

    uint32_t f7 = (insn >> 25) & 0x7F;
    uint32_t rs2_field = (insn >> 20) & 0x1F;
    uint32_t imm12 = (insn >> 20) & 0xFFF;

    if (f3 == 1 && f7 == 0x30) {
        switch (rs2_field) {
        case 0: { // CLZ
            int n = __clzll(a);
            result = a ? n : 64;
            break;
        }
        case 1: { // CTZ
            int n = __ffsll(a);
            result = n ? n - 1 : 64;
            break;
        }
        case 2:  result = __popcll(a); break;                    // CPOP
        case 4:  result = (int64_t)(int8_t)(uint8_t)a; break;   // SEXT.B
        case 5:  result = (int64_t)(int16_t)(uint16_t)a; break; // SEXT.H
        default: return false;
        }
    } else if (f3 == 5 && f7 == 0x30) {
        // RORI: rotate right immediate
        uint32_t shamt = (insn >> 20) & 0x3F;
        result = (a >> shamt) | (a << ((64 - shamt) & 63));
    } else if (f3 == 5 && imm12 == 0x287) {
        // ORC.B: bitwise OR-combine bytes
        result = 0;
        for (int i = 0; i < 8; i++) {
            uint8_t byte = (a >> (i * 8)) & 0xFF;
            result |= (uint64_t)(byte ? 0xFF : 0x00) << (i * 8);
        }
    } else if (f3 == 5 && imm12 == 0x6B8) {
        // REV8: byte-reverse
        uint32_t lo = (uint32_t)a, hi = (uint32_t)(a >> 32);
        lo = ((lo >> 24) & 0xFF) | ((lo >> 8) & 0xFF00) | ((lo << 8) & 0xFF0000) | ((lo << 24) & 0xFF000000);
        hi = ((hi >> 24) & 0xFF) | ((hi >> 8) & 0xFF00) | ((hi << 8) & 0xFF0000) | ((hi << 24) & 0xFF000000);
        result = ((uint64_t)lo << 32) | hi;
    } else {
        return false;
    }

    if (d) hart->x[d] = result;
    return true;
}

__device__ bool try_exec_zbb_imm32(HartState* hart, uint32_t insn) {
    uint32_t f3 = funct3(insn);
    uint32_t d = rd(insn);
    uint32_t a = (uint32_t)hart->x[rs1(insn)];
    int32_t result;

    uint32_t f7 = (insn >> 25) & 0x7F;
    uint32_t rs2_field = (insn >> 20) & 0x1F;

    if (f3 == 1 && f7 == 0x30) {
        switch (rs2_field) {
        case 0:  result = a ? __clz(a) : 32; break;                // CLZ.W
        case 1:  result = a ? (__ffs(a) - 1) : 32; break;          // CTZ.W
        case 2:  result = __popc(a); break;                          // CPOP.W
        default: return false;
        }
    } else if (f3 == 5 && f7 == 0x30) {
        // RORI.W
        uint32_t shamt = (insn >> 20) & 0x1F;
        result = (int32_t)((a >> shamt) | (a << ((32 - shamt) & 31)));
    } else {
        return false;
    }

    if (d) hart->x[d] = (int64_t)result;
    return true;
}

// ---- Zbs: single-bit operations ----

__device__ bool try_exec_zbs_op(HartState* hart, uint32_t insn) {
    uint32_t f7 = funct7(insn), f3 = funct3(insn);
    uint32_t d = rd(insn);
    uint64_t a = hart->x[rs1(insn)];
    uint64_t shamt = hart->x[rs2(insn)] & 63;
    uint64_t result;

    if (f3 == 1) {
        if (f7 == 0x24)      result = a & ~(1ULL << shamt);        // BCLR
        else if (f7 == 0x34) result = a ^ (1ULL << shamt);         // BINV
        else if (f7 == 0x14) result = a | (1ULL << shamt);         // BSET
        else return false;
    } else if (f3 == 5 && f7 == 0x24) {
        result = (a >> shamt) & 1;                                   // BEXT
    } else return false;

    if (d) hart->x[d] = result;
    return true;
}

__device__ bool try_exec_zbs_imm(HartState* hart, uint32_t insn) {
    uint32_t f3 = funct3(insn);
    uint32_t d = rd(insn);
    uint64_t a = hart->x[rs1(insn)];
    uint32_t shamt = (insn >> 20) & 0x3F;
    uint32_t f7_top = (insn >> 26) & 0x3F;  // bits 31:26
    uint64_t result;

    if (f3 == 1) {
        if (f7_top == 0x12)      result = a & ~(1ULL << shamt);    // BCLRI
        else if (f7_top == 0x1A) result = a ^ (1ULL << shamt);     // BINVI
        else if (f7_top == 0x0A) result = a | (1ULL << shamt);     // BSETI
        else return false;
    } else if (f3 == 5 && f7_top == 0x12) {
        result = (a >> shamt) & 1;                                   // BEXTI
    } else return false;

    if (d) hart->x[d] = result;
    return true;
}

// ---- Zicboz: cache block zero ----
__device__ bool try_exec_cbo(HartState* hart, Machine* m, uint32_t insn) {
    uint32_t f3 = funct3(insn);
    if (f3 != 2) return false;  // CBO instructions use funct3=2 in MISC-MEM

    uint32_t f12 = (insn >> 20) & 0xFFF;
    uint64_t addr = hart->x[rs1(insn)];

    switch (f12) {
    case 0: // CBO.INVAL — invalidate cache block (NOP for emulator)
    case 1: // CBO.CLEAN — clean cache block (NOP)
    case 2: // CBO.FLUSH — flush cache block (NOP)
        return true;
    case 4: { // CBO.ZERO — zero 64 bytes at aligned address
        addr &= ~63ULL;
        for (int i = 0; i < 64; i += 8)
            mem_store(hart, m, addr + i, 8, 0);
        return true;
    }
    default: return false;
    }
}
