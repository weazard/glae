#pragma once
#include "../core/hart.cuh"
#include "../priv/trap.cuh"
#include <math.h>

// Double-precision helpers
__device__ __forceinline__ double as_double(uint64_t v) {
    double d; memcpy(&d, &v, 8); return d;
}
__device__ __forceinline__ uint64_t as_u64(double d) {
    uint64_t v; memcpy(&v, &d, 8); return v;
}

// ============================================================
// D Extension — Double-Precision Arithmetic (opcode 0x53)
// ============================================================
__device__ bool exec_op_fp_d(HartState* hart, uint32_t insn) {
    if (hart->fs() == 0) {
        take_trap(hart, EXC_ILLEGAL_INSN, hart->pc, insn);
        return true;
    }
    uint32_t d = rd(insn);
    uint32_t f7 = funct7(insn);
    double a = as_double(hart->f[rs1(insn)]);
    double b = as_double(hart->f[rs2(insn)]);
    uint32_t rm = funct3(insn);

    hart->mark_fs_dirty();

    switch (f7) {
    case 0x01: hart->f[d] = as_u64(a + b); return false; // FADD.D
    case 0x05: hart->f[d] = as_u64(a - b); return false; // FSUB.D
    case 0x09: hart->f[d] = as_u64(a * b); return false; // FMUL.D
    case 0x0D: hart->f[d] = as_u64(a / b); return false; // FDIV.D
    case 0x2D: hart->f[d] = as_u64(sqrt(a)); return false; // FSQRT.D
    case 0x11: // FSGNJ.D / FSGNJN.D / FSGNJX.D
        switch (rm) {
        case 0: { // FSGNJ.D
            uint64_t ua = as_u64(a) & 0x7FFFFFFFFFFFFFFFULL;
            uint64_t ub = as_u64(b) & 0x8000000000000000ULL;
            hart->f[d] = ua | ub;
            return false;
        }
        case 1: { // FSGNJN.D
            uint64_t ua = as_u64(a) & 0x7FFFFFFFFFFFFFFFULL;
            uint64_t ub = as_u64(b) ^ 0x8000000000000000ULL;
            hart->f[d] = ua | (ub & 0x8000000000000000ULL);
            return false;
        }
        case 2: { // FSGNJX.D
            hart->f[d] = as_u64(a) ^ (as_u64(b) & 0x8000000000000000ULL);
            return false;
        }
        }
        break;
    case 0x15: // FMIN.D / FMAX.D
        if (rm == 0)      hart->f[d] = as_u64(fmin(a, b));
        else if (rm == 1) hart->f[d] = as_u64(fmax(a, b));
        return false;
    case 0x61: { // FCVT.W.D / FCVT.WU.D / FCVT.L.D / FCVT.LU.D
        uint32_t s2 = rs2(insn);
        switch (s2) {
        case 0: if (d) hart->x[d] = (int64_t)(int32_t)(int)a; return false;
        case 1: if (d) hart->x[d] = (int64_t)(int32_t)(uint32_t)a; return false;
        case 2: if (d) hart->x[d] = (int64_t)llrint(a); return false;
        case 3: if (d) hart->x[d] = (uint64_t)(unsigned long long)llrint(fabs(a)); return false;
        }
        break;
    }
    case 0x69: { // FCVT.D.W / FCVT.D.WU / FCVT.D.L / FCVT.D.LU
        uint32_t s2 = rs2(insn);
        switch (s2) {
        case 0: hart->f[d] = as_u64((double)(int32_t)hart->x[rs1(insn)]); return false;
        case 1: hart->f[d] = as_u64((double)(uint32_t)hart->x[rs1(insn)]); return false;
        case 2: hart->f[d] = as_u64((double)(int64_t)hart->x[rs1(insn)]); return false;
        case 3: hart->f[d] = as_u64((double)(uint64_t)hart->x[rs1(insn)]); return false;
        }
        break;
    }
    case 0x71: // FMV.X.D / FCLASS.D
        if (rm == 0) { // FMV.X.D
            if (d) hart->x[d] = hart->f[rs1(insn)];
        } else if (rm == 1) { // FCLASS.D
            uint64_t bv = as_u64(a);
            bool sign = (bv >> 63) & 1;
            uint32_t exp = (bv >> 52) & 0x7FF;
            uint64_t frac = bv & 0xFFFFFFFFFFFFFULL;
            uint32_t cls = 0;
            if (exp == 0x7FF && frac) cls = (frac & (1ULL << 51)) ? (1 << 9) : (1 << 8);
            else if (exp == 0x7FF) cls = sign ? (1 << 0) : (1 << 7);
            else if (exp == 0 && frac == 0) cls = sign ? (1 << 3) : (1 << 4);
            else if (exp == 0) cls = sign ? (1 << 2) : (1 << 5);
            else cls = sign ? (1 << 1) : (1 << 6);
            if (d) hart->x[d] = cls;
        }
        return false;
    case 0x79: // FMV.D.X
        if (rm == 0) hart->f[d] = hart->x[rs1(insn)];
        return false;
    case 0x51: // FEQ.D / FLT.D / FLE.D
        switch (rm) {
        case 2: if (d) hart->x[d] = (a == b) ? 1 : 0; return false;
        case 1: if (d) hart->x[d] = (a < b) ? 1 : 0; return false;
        case 0: if (d) hart->x[d] = (a <= b) ? 1 : 0; return false;
        }
        break;
    case 0x21: // FCVT.D.S
        if (rs2(insn) == 0) {
            float fv = unbox_f(hart->f[rs1(insn)]);
            hart->f[d] = as_u64((double)fv);
            return false;
        }
        break;
    }

    take_trap(hart, EXC_ILLEGAL_INSN, hart->pc, insn);
    return true;
}

// FMA group (FMADD.D, FMSUB.D, FNMSUB.D, FNMADD.D)
__device__ bool exec_fma_d(HartState* hart, uint32_t insn, uint32_t op) {
    if (hart->fs() == 0) {
        take_trap(hart, EXC_ILLEGAL_INSN, hart->pc, insn);
        return true;
    }
    uint32_t d = rd(insn);
    double a = as_double(hart->f[rs1(insn)]);
    double b = as_double(hart->f[rs2(insn)]);
    double c = as_double(hart->f[rs3(insn)]);
    double result;
    hart->mark_fs_dirty();

    switch (op) {
    case OP_FMADD:  result = fma(a, b, c);   break;
    case OP_FMSUB:  result = fma(a, b, -c);  break;
    case OP_FNMSUB: result = fma(-a, b, c);  break;
    case OP_FNMADD: result = fma(-a, b, -c); break;
    default: result = 0;
    }

    hart->f[d] = as_u64(result);
    return false;
}
