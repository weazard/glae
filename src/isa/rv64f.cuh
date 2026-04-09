#pragma once
#include "../core/hart.cuh"
#include "../priv/mmu.cuh"
#include "../priv/trap.cuh"
#include <math.h>

// NaN-boxing helpers for single-precision in 64-bit register
__device__ __forceinline__ float unbox_f(uint64_t v) {
    if ((v & 0xFFFFFFFF00000000ULL) == 0xFFFFFFFF00000000ULL)
        return __uint_as_float((uint32_t)v);
    // Not properly NaN-boxed: canonical NaN
    return __uint_as_float(0x7FC00000);
}

__device__ __forceinline__ uint64_t box_f(float f) {
    return 0xFFFFFFFF00000000ULL | (uint64_t)__float_as_uint(f);
}

// ============================================================
// FP Load/Store
// ============================================================
__device__ bool exec_load_fp(HartState* hart, Machine* m, uint32_t insn) {
    if (hart->fs() == 0) {
        take_trap(hart, EXC_ILLEGAL_INSN, hart->pc, insn);
        return true;
    }
    uint32_t d = rd(insn);
    uint64_t addr = hart->x[rs1(insn)] + imm_i(insn);
    uint32_t f = funct3(insn);
    uint64_t val = 0;

    if (f == 2) { // FLW
        if (!mem_load(hart, m, addr, 4, &val)) return true;
        hart->f[d] = box_f(__uint_as_float((uint32_t)val));
    } else if (f == 3) { // FLD
        if (!mem_load(hart, m, addr, 8, &val)) return true;
        hart->f[d] = val;
    } else {
        take_trap(hart, EXC_ILLEGAL_INSN, hart->pc, insn);
        return true;
    }
    hart->mark_fs_dirty();
    return false;
}

__device__ bool exec_store_fp(HartState* hart, Machine* m, uint32_t insn) {
    if (hart->fs() == 0) {
        take_trap(hart, EXC_ILLEGAL_INSN, hart->pc, insn);
        return true;
    }
    uint64_t addr = hart->x[rs1(insn)] + imm_s(insn);
    uint32_t f = funct3(insn);
    uint32_t s2 = rs2(insn);

    if (f == 2) { // FSW
        uint32_t val = (uint32_t)hart->f[s2];
        if (!mem_store(hart, m, addr, 4, val)) return true;
    } else if (f == 3) { // FSD
        if (!mem_store(hart, m, addr, 8, hart->f[s2])) return true;
    } else {
        take_trap(hart, EXC_ILLEGAL_INSN, hart->pc, insn);
        return true;
    }
    return false;
}

// ============================================================
// F Extension — Single-Precision Arithmetic (opcode 0x53)
// ============================================================
__device__ bool exec_op_fp_s(HartState* hart, uint32_t insn) {
    if (hart->fs() == 0) {
        take_trap(hart, EXC_ILLEGAL_INSN, hart->pc, insn);
        return true;
    }
    uint32_t d = rd(insn);
    uint32_t f7 = funct7(insn);
    float a = unbox_f(hart->f[rs1(insn)]);
    float b = unbox_f(hart->f[rs2(insn)]);
    uint32_t rm = funct3(insn);

    hart->mark_fs_dirty();

    switch (f7) {
    case 0x00: hart->f[d] = box_f(a + b); return false; // FADD.S
    case 0x04: hart->f[d] = box_f(a - b); return false; // FSUB.S
    case 0x08: hart->f[d] = box_f(a * b); return false; // FMUL.S
    case 0x0C: hart->f[d] = box_f(a / b); return false; // FDIV.S
    case 0x2C: // FSQRT.S
        hart->f[d] = box_f(sqrtf(a));
        return false;
    case 0x10: // FSGNJ / FSGNJN / FSGNJX
        switch (rm) {
        case 0: { // FSGNJ.S
            uint32_t ua = __float_as_uint(a) & 0x7FFFFFFF;
            uint32_t ub = __float_as_uint(b) & 0x80000000;
            hart->f[d] = box_f(__uint_as_float(ua | ub));
            return false;
        }
        case 1: { // FSGNJN.S
            uint32_t ua = __float_as_uint(a) & 0x7FFFFFFF;
            uint32_t ub = __float_as_uint(b) ^ 0x80000000;
            hart->f[d] = box_f(__uint_as_float(ua | (ub & 0x80000000)));
            return false;
        }
        case 2: { // FSGNJX.S
            uint32_t ua = __float_as_uint(a);
            uint32_t ub = __float_as_uint(b);
            hart->f[d] = box_f(__uint_as_float(ua ^ (ub & 0x80000000)));
            return false;
        }
        }
        break;
    case 0x14: // FMIN.S / FMAX.S
        if (rm == 0)      hart->f[d] = box_f(fminf(a, b));
        else if (rm == 1) hart->f[d] = box_f(fmaxf(a, b));
        return false;
    case 0x60: { // FCVT.W.S / FCVT.WU.S / FCVT.L.S / FCVT.LU.S
        uint32_t s2 = rs2(insn);
        switch (s2) {
        case 0: if (d) hart->x[d] = (int64_t)(int32_t)__float2int_rn(a); return false; // FCVT.W.S
        case 1: if (d) hart->x[d] = (int64_t)(int32_t)(uint32_t)__float2uint_rn(a); return false; // FCVT.WU.S
        case 2: if (d) hart->x[d] = (int64_t)llrintf(a); return false; // FCVT.L.S
        case 3: if (d) hart->x[d] = (uint64_t)(unsigned long long)llrintf(fabsf(a)); return false; // FCVT.LU.S
        }
        break;
    }
    case 0x68: { // FCVT.S.W / FCVT.S.WU / FCVT.S.L / FCVT.S.LU
        uint32_t s2 = rs2(insn);
        switch (s2) {
        case 0: hart->f[d] = box_f((float)(int32_t)hart->x[rs1(insn)]); return false;
        case 1: hart->f[d] = box_f((float)(uint32_t)hart->x[rs1(insn)]); return false;
        case 2: hart->f[d] = box_f((float)(int64_t)hart->x[rs1(insn)]); return false;
        case 3: hart->f[d] = box_f((float)(uint64_t)hart->x[rs1(insn)]); return false;
        }
        break;
    }
    case 0x70: // FMV.X.W / FCLASS.S
        if (rm == 0) { // FMV.X.W
            if (d) hart->x[d] = (int64_t)(int32_t)(uint32_t)hart->f[rs1(insn)];
        } else if (rm == 1) { // FCLASS.S
            uint32_t bits_val = __float_as_uint(a);
            bool sign = (bits_val >> 31) & 1;
            uint32_t exp = (bits_val >> 23) & 0xFF;
            uint32_t frac = bits_val & 0x7FFFFF;
            uint32_t cls = 0;
            if (exp == 0xFF && frac) cls = (frac & 0x400000) ? (1 << 9) : (1 << 8); // qNaN/sNaN
            else if (exp == 0xFF) cls = sign ? (1 << 0) : (1 << 7); // ±inf
            else if (exp == 0 && frac == 0) cls = sign ? (1 << 3) : (1 << 4); // ±0
            else if (exp == 0) cls = sign ? (1 << 2) : (1 << 5); // subnormal
            else cls = sign ? (1 << 1) : (1 << 6); // normal
            if (d) hart->x[d] = cls;
        }
        return false;
    case 0x78: // FMV.W.X
        if (rm == 0) {
            hart->f[d] = box_f(__uint_as_float((uint32_t)hart->x[rs1(insn)]));
        }
        return false;
    case 0x50: // FEQ.S / FLT.S / FLE.S
        switch (rm) {
        case 2: if (d) hart->x[d] = (a == b) ? 1 : 0; return false; // FEQ.S
        case 1: if (d) hart->x[d] = (a < b) ? 1 : 0; return false;  // FLT.S
        case 0: if (d) hart->x[d] = (a <= b) ? 1 : 0; return false; // FLE.S
        }
        break;
    case 0x20: // FCVT.S.D
        if (rs2(insn) == 1) {
            double dv;
            memcpy(&dv, &hart->f[rs1(insn)], 8);
            hart->f[d] = box_f((float)dv);
            return false;
        }
        break;
    }

    take_trap(hart, EXC_ILLEGAL_INSN, hart->pc, insn);
    return true;
}

// FMA group (FMADD.S, FMSUB.S, FNMSUB.S, FNMADD.S)
__device__ bool exec_fma_s(HartState* hart, uint32_t insn, uint32_t op) {
    if (hart->fs() == 0) {
        take_trap(hart, EXC_ILLEGAL_INSN, hart->pc, insn);
        return true;
    }
    uint32_t d = rd(insn);
    float a = unbox_f(hart->f[rs1(insn)]);
    float b = unbox_f(hart->f[rs2(insn)]);
    float c = unbox_f(hart->f[rs3(insn)]);
    float result;
    hart->mark_fs_dirty();

    switch (op) {
    case OP_FMADD:  result = fmaf(a, b, c);   break; //  a*b+c
    case OP_FMSUB:  result = fmaf(a, b, -c);  break; //  a*b-c
    case OP_FNMSUB: result = fmaf(-a, b, c);  break; // -a*b+c
    case OP_FNMADD: result = fmaf(-a, b, -c); break; // -a*b-c
    default: result = 0;
    }

    hart->f[d] = box_f(result);
    return false;
}
