#pragma once
#include "../core/hart.cuh"

// Decompress a 16-bit compressed instruction to its 32-bit equivalent.
// Returns 0 for illegal/reserved encodings.
__device__ uint32_t decompress_c(uint16_t ci) {
    uint32_t q = ci & 3;        // quadrant
    uint32_t f3 = (ci >> 13) & 7; // funct3

    switch (q) {
    // ---- Quadrant 0 ----
    case 0:
        switch (f3) {
        case 0: { // C.ADDI4SPN
            uint32_t nzimm = ((ci >> 1) & 0x40) | ((ci >> 7) & 0x30) |
                             ((ci >> 1) & 0x8) | ((ci >> 4) & 0x4);
            // nzimm[5:4|9:6|2|3]
            nzimm = (bits(ci, 12, 11) << 4) | (bits(ci, 10, 7) << 6) |
                    (bits(ci, 6, 6) << 2) | (bits(ci, 5, 5) << 3);
            if (nzimm == 0) return 0; // reserved
            uint32_t rrd = 8 + bits(ci, 4, 2);
            // ADDI rd', x2, nzimm
            return (nzimm << 20) | (2 << 15) | (0 << 12) | (rrd << 7) | OP_OP_IMM;
        }
        case 1: { // C.FLD
            uint32_t rrd = 8 + bits(ci, 4, 2);
            uint32_t rrs = 8 + bits(ci, 9, 7);
            uint32_t off = (bits(ci, 12, 10) << 3) | (bits(ci, 6, 5) << 6);
            return (off << 20) | (rrs << 15) | (3 << 12) | (rrd << 7) | OP_LOAD_FP;
        }
        case 2: { // C.LW
            uint32_t rrd = 8 + bits(ci, 4, 2);
            uint32_t rrs = 8 + bits(ci, 9, 7);
            uint32_t off = (bits(ci, 12, 10) << 3) | (bits(ci, 6, 6) << 2) | (bits(ci, 5, 5) << 6);
            return (off << 20) | (rrs << 15) | (2 << 12) | (rrd << 7) | OP_LOAD;
        }
        case 3: { // C.LD (RV64)
            uint32_t rrd = 8 + bits(ci, 4, 2);
            uint32_t rrs = 8 + bits(ci, 9, 7);
            uint32_t off = (bits(ci, 12, 10) << 3) | (bits(ci, 6, 5) << 6);
            return (off << 20) | (rrs << 15) | (3 << 12) | (rrd << 7) | OP_LOAD;
        }
        case 5: { // C.FSD
            uint32_t rrs2 = 8 + bits(ci, 4, 2);
            uint32_t rrs1 = 8 + bits(ci, 9, 7);
            uint32_t off = (bits(ci, 12, 10) << 3) | (bits(ci, 6, 5) << 6);
            uint32_t imm115 = off >> 5;
            uint32_t imm40 = off & 0x1F;
            return (imm115 << 25) | (rrs2 << 20) | (rrs1 << 15) | (3 << 12) | (imm40 << 7) | OP_STORE_FP;
        }
        case 6: { // C.SW
            uint32_t rrs2 = 8 + bits(ci, 4, 2);
            uint32_t rrs1 = 8 + bits(ci, 9, 7);
            uint32_t off = (bits(ci, 12, 10) << 3) | (bits(ci, 6, 6) << 2) | (bits(ci, 5, 5) << 6);
            uint32_t imm115 = off >> 5;
            uint32_t imm40 = off & 0x1F;
            return (imm115 << 25) | (rrs2 << 20) | (rrs1 << 15) | (2 << 12) | (imm40 << 7) | OP_STORE;
        }
        case 7: { // C.SD (RV64)
            uint32_t rrs2 = 8 + bits(ci, 4, 2);
            uint32_t rrs1 = 8 + bits(ci, 9, 7);
            uint32_t off = (bits(ci, 12, 10) << 3) | (bits(ci, 6, 5) << 6);
            uint32_t imm115 = off >> 5;
            uint32_t imm40 = off & 0x1F;
            return (imm115 << 25) | (rrs2 << 20) | (rrs1 << 15) | (3 << 12) | (imm40 << 7) | OP_STORE;
        }
        default: return 0;
        }

    // ---- Quadrant 1 ----
    case 1:
        switch (f3) {
        case 0: { // C.NOP / C.ADDI
            uint32_t rrd = bits(ci, 11, 7);
            int32_t imm = (int32_t)(((ci >> 7) & 0x20) | bits(ci, 6, 2));
            if (imm & 0x20) imm |= ~0x3F; // sign extend
            return ((uint32_t)imm << 20) | (rrd << 15) | (0 << 12) | (rrd << 7) | OP_OP_IMM;
        }
        case 1: { // C.ADDIW (RV64)
            uint32_t rrd = bits(ci, 11, 7);
            if (rrd == 0) return 0; // reserved
            int32_t imm = (int32_t)(((ci >> 7) & 0x20) | bits(ci, 6, 2));
            if (imm & 0x20) imm |= ~0x3F;
            return ((uint32_t)imm << 20) | (rrd << 15) | (0 << 12) | (rrd << 7) | OP_OP_IMM_32;
        }
        case 2: { // C.LI → ADDI rd, x0, imm
            uint32_t rrd = bits(ci, 11, 7);
            int32_t imm = (int32_t)(((ci >> 7) & 0x20) | bits(ci, 6, 2));
            if (imm & 0x20) imm |= ~0x3F;
            return ((uint32_t)imm << 20) | (0 << 15) | (0 << 12) | (rrd << 7) | OP_OP_IMM;
        }
        case 3: { // C.ADDI16SP / C.LUI
            uint32_t rrd = bits(ci, 11, 7);
            if (rrd == 2) {
                // C.ADDI16SP: ADDI x2, x2, nzimm
                int32_t imm = (int32_t)(
                    (bits(ci, 12, 12) << 9) | (bits(ci, 6, 6) << 4) |
                    (bits(ci, 5, 5) << 6) | (bits(ci, 4, 3) << 7) |
                    (bits(ci, 2, 2) << 5));
                if (imm & 0x200) imm |= ~0x3FF; // sign extend from bit 9
                if (imm == 0) return 0;
                return ((uint32_t)imm << 20) | (2 << 15) | (0 << 12) | (2 << 7) | OP_OP_IMM;
            } else {
                // C.LUI
                int32_t imm = (int32_t)(((ci >> 7) & 0x20) | bits(ci, 6, 2));
                if (imm & 0x20) imm |= ~0x3F;
                if (imm == 0) return 0;
                uint32_t uimm = ((uint32_t)imm << 12);
                return (uimm & 0xFFFFF000) | (rrd << 7) | OP_LUI;
            }
        }
        case 4: { // C.SRLI / C.SRAI / C.ANDI / C.SUB/XOR/OR/AND/SUBW/ADDW
            uint32_t rrd = 8 + bits(ci, 9, 7);
            uint32_t f2 = bits(ci, 11, 10);
            switch (f2) {
            case 0: { // C.SRLI
                uint32_t shamt = (bits(ci, 12, 12) << 5) | bits(ci, 6, 2);
                return (0x00 << 25) | (shamt << 20) | (rrd << 15) | (5 << 12) | (rrd << 7) | OP_OP_IMM;
            }
            case 1: { // C.SRAI
                uint32_t shamt = (bits(ci, 12, 12) << 5) | bits(ci, 6, 2);
                return (0x20 << 25) | (shamt << 20) | (rrd << 15) | (5 << 12) | (rrd << 7) | OP_OP_IMM;
            }
            case 2: { // C.ANDI
                int32_t imm = (int32_t)(((ci >> 7) & 0x20) | bits(ci, 6, 2));
                if (imm & 0x20) imm |= ~0x3F;
                return ((uint32_t)imm << 20) | (rrd << 15) | (7 << 12) | (rrd << 7) | OP_OP_IMM;
            }
            case 3: {
                uint32_t rrs2 = 8 + bits(ci, 4, 2);
                uint32_t f1 = bits(ci, 12, 12);
                uint32_t f2b = bits(ci, 6, 5);
                if (f1 == 0) {
                    switch (f2b) {
                    case 0: // C.SUB
                        return (0x20 << 25) | (rrs2 << 20) | (rrd << 15) | (0 << 12) | (rrd << 7) | OP_OP;
                    case 1: // C.XOR
                        return (0x00 << 25) | (rrs2 << 20) | (rrd << 15) | (4 << 12) | (rrd << 7) | OP_OP;
                    case 2: // C.OR
                        return (0x00 << 25) | (rrs2 << 20) | (rrd << 15) | (6 << 12) | (rrd << 7) | OP_OP;
                    case 3: // C.AND
                        return (0x00 << 25) | (rrs2 << 20) | (rrd << 15) | (7 << 12) | (rrd << 7) | OP_OP;
                    }
                } else {
                    switch (f2b) {
                    case 0: // C.SUBW
                        return (0x20 << 25) | (rrs2 << 20) | (rrd << 15) | (0 << 12) | (rrd << 7) | OP_OP_32;
                    case 1: // C.ADDW
                        return (0x00 << 25) | (rrs2 << 20) | (rrd << 15) | (0 << 12) | (rrd << 7) | OP_OP_32;
                    default: return 0;
                    }
                }
                return 0;
            }
            }
            return 0;
        }
        case 5: { // C.J → JAL x0, offset
            // offset[11|4|9:8|10|6|7|3:1|5]
            int32_t off = (int32_t)(
                (bits(ci, 12, 12) << 11) | (bits(ci, 11, 11) << 4) |
                (bits(ci, 10, 9) << 8) | (bits(ci, 8, 8) << 10) |
                (bits(ci, 7, 7) << 6) | (bits(ci, 6, 6) << 7) |
                (bits(ci, 5, 3) << 1) | (bits(ci, 2, 2) << 5));
            if (off & 0x800) off |= ~0xFFF;
            // Encode as JAL x0, off
            uint32_t imm20 = ((off >> 20) & 1) << 31;
            uint32_t imm101 = ((off >> 1) & 0x3FF) << 21;
            uint32_t imm11 = ((off >> 11) & 1) << 20;
            uint32_t imm1912 = ((off >> 12) & 0xFF) << 12;
            return imm20 | imm101 | imm11 | imm1912 | (0 << 7) | OP_JAL;
        }
        case 6: { // C.BEQZ
            uint32_t rrs = 8 + bits(ci, 9, 7);
            int32_t off = (int32_t)(
                (bits(ci, 12, 12) << 8) | (bits(ci, 11, 10) << 3) |
                (bits(ci, 6, 5) << 6) | (bits(ci, 4, 3) << 1) |
                (bits(ci, 2, 2) << 5));
            if (off & 0x100) off |= ~0x1FF;
            // BEQ rs', x0, off
            uint32_t imm12 = (off >> 12) & 1;
            uint32_t imm105 = (off >> 5) & 0x3F;
            uint32_t imm41 = (off >> 1) & 0xF;
            uint32_t imm11 = (off >> 11) & 1;
            return (imm12 << 31) | (imm105 << 25) | (0 << 20) | (rrs << 15) |
                   (0 << 12) | (imm41 << 8) | (imm11 << 7) | OP_BRANCH;
        }
        case 7: { // C.BNEZ
            uint32_t rrs = 8 + bits(ci, 9, 7);
            int32_t off = (int32_t)(
                (bits(ci, 12, 12) << 8) | (bits(ci, 11, 10) << 3) |
                (bits(ci, 6, 5) << 6) | (bits(ci, 4, 3) << 1) |
                (bits(ci, 2, 2) << 5));
            if (off & 0x100) off |= ~0x1FF;
            uint32_t imm12 = (off >> 12) & 1;
            uint32_t imm105 = (off >> 5) & 0x3F;
            uint32_t imm41 = (off >> 1) & 0xF;
            uint32_t imm11 = (off >> 11) & 1;
            return (imm12 << 31) | (imm105 << 25) | (0 << 20) | (rrs << 15) |
                   (1 << 12) | (imm41 << 8) | (imm11 << 7) | OP_BRANCH;
        }
        }
        return 0;

    // ---- Quadrant 2 ----
    case 2:
        switch (f3) {
        case 0: { // C.SLLI
            uint32_t rrd = bits(ci, 11, 7);
            uint32_t shamt = (bits(ci, 12, 12) << 5) | bits(ci, 6, 2);
            if (rrd == 0) return 0;
            return (shamt << 20) | (rrd << 15) | (1 << 12) | (rrd << 7) | OP_OP_IMM;
        }
        case 1: { // C.FLDSP
            uint32_t rrd = bits(ci, 11, 7);
            uint32_t off = (bits(ci, 12, 12) << 5) | (bits(ci, 6, 5) << 3) | (bits(ci, 4, 2) << 6);
            return (off << 20) | (2 << 15) | (3 << 12) | (rrd << 7) | OP_LOAD_FP;
        }
        case 2: { // C.LWSP
            uint32_t rrd = bits(ci, 11, 7);
            if (rrd == 0) return 0;
            uint32_t off = (bits(ci, 12, 12) << 5) | (bits(ci, 6, 4) << 2) | (bits(ci, 3, 2) << 6);
            return (off << 20) | (2 << 15) | (2 << 12) | (rrd << 7) | OP_LOAD;
        }
        case 3: { // C.LDSP (RV64)
            uint32_t rrd = bits(ci, 11, 7);
            if (rrd == 0) return 0;
            uint32_t off = (bits(ci, 12, 12) << 5) | (bits(ci, 6, 5) << 3) | (bits(ci, 4, 2) << 6);
            return (off << 20) | (2 << 15) | (3 << 12) | (rrd << 7) | OP_LOAD;
        }
        case 4: {
            uint32_t rrd = bits(ci, 11, 7);
            uint32_t rrs2 = bits(ci, 6, 2);
            uint32_t b12 = bits(ci, 12, 12);
            if (b12 == 0) {
                if (rrs2 == 0) {
                    // C.JR → JALR x0, rs1, 0
                    if (rrd == 0) return 0;
                    return (0 << 20) | (rrd << 15) | (0 << 12) | (0 << 7) | OP_JALR;
                } else {
                    // C.MV → ADD rd, x0, rs2
                    return (0 << 25) | (rrs2 << 20) | (0 << 15) | (0 << 12) | (rrd << 7) | OP_OP;
                }
            } else {
                if (rrs2 == 0 && rrd == 0) {
                    // C.EBREAK
                    return 0x00100073; // EBREAK encoding
                } else if (rrs2 == 0) {
                    // C.JALR → JALR x1, rs1, 0
                    return (0 << 20) | (rrd << 15) | (0 << 12) | (1 << 7) | OP_JALR;
                } else {
                    // C.ADD → ADD rd, rd, rs2
                    return (0 << 25) | (rrs2 << 20) | (rrd << 15) | (0 << 12) | (rrd << 7) | OP_OP;
                }
            }
        }
        case 5: { // C.FSDSP
            uint32_t rrs2 = bits(ci, 6, 2);
            uint32_t off = (bits(ci, 12, 10) << 3) | (bits(ci, 9, 7) << 6);
            uint32_t imm115 = off >> 5;
            uint32_t imm40 = off & 0x1F;
            return (imm115 << 25) | (rrs2 << 20) | (2 << 15) | (3 << 12) | (imm40 << 7) | OP_STORE_FP;
        }
        case 6: { // C.SWSP
            uint32_t rrs2 = bits(ci, 6, 2);
            uint32_t off = (bits(ci, 12, 9) << 2) | (bits(ci, 8, 7) << 6);
            uint32_t imm115 = off >> 5;
            uint32_t imm40 = off & 0x1F;
            return (imm115 << 25) | (rrs2 << 20) | (2 << 15) | (2 << 12) | (imm40 << 7) | OP_STORE;
        }
        case 7: { // C.SDSP (RV64)
            uint32_t rrs2 = bits(ci, 6, 2);
            uint32_t off = (bits(ci, 12, 10) << 3) | (bits(ci, 9, 7) << 6);
            uint32_t imm115 = off >> 5;
            uint32_t imm40 = off & 0x1F;
            return (imm115 << 25) | (rrs2 << 20) | (2 << 15) | (3 << 12) | (imm40 << 7) | OP_STORE;
        }
        }
        return 0;

    default: return 0;
    }
}
