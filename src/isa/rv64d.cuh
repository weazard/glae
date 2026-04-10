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

// Rounding-mode-aware double arithmetic using CUDA intrinsics
__device__ __forceinline__ double dadd_rm(double a, double b, uint32_t rm) {
    switch (rm) { case 1: return __dadd_rz(a,b); case 2: return __dadd_rd(a,b);
                  case 3: return __dadd_ru(a,b); default: return __dadd_rn(a,b); }
}
__device__ __forceinline__ double dmul_rm(double a, double b, uint32_t rm) {
    switch (rm) { case 1: return __dmul_rz(a,b); case 2: return __dmul_rd(a,b);
                  case 3: return __dmul_ru(a,b); default: return __dmul_rn(a,b); }
}
__device__ __forceinline__ double ddiv_rm(double a, double b, uint32_t rm) {
    switch (rm) { case 1: return __ddiv_rz(a,b); case 2: return __ddiv_rd(a,b);
                  case 3: return __ddiv_ru(a,b); default: return __ddiv_rn(a,b); }
}
__device__ __forceinline__ double roundd_rm(double a, uint32_t rm) {
    switch (rm) { case 1: return trunc(a); case 2: return floor(a);
                  case 3: return ceil(a); case 4: return round(a);
                  default: return rint(a); }
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
    uint32_t erm = resolve_rm(hart, rm);

    switch (f7) {
    case 0x01: hart->f[d] = as_u64(dadd_rm(a, b, erm)); return false;  // FADD.D
    case 0x05: hart->f[d] = as_u64(dadd_rm(a, -b, erm)); return false; // FSUB.D
    case 0x09: hart->f[d] = as_u64(dmul_rm(a, b, erm)); return false;  // FMUL.D
    case 0x0D: // FDIV.D
        if (b == 0.0 && isfinite(a) && a != 0.0) hart->fcsr |= FFLAG_DZ;
        hart->f[d] = as_u64(ddiv_rm(a, b, erm));
        return false;
    case 0x2D: // FSQRT.D
        if (a < 0.0 && !isnan(a)) hart->fcsr |= FFLAG_NV;
        hart->f[d] = as_u64(sqrt(a));
        return false;
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
        if (is_snan_d(as_u64(a)) || is_snan_d(as_u64(b))) hart->fcsr |= FFLAG_NV;
        if (rm == 0)      hart->f[d] = as_u64(fmin(a, b));
        else if (rm == 1) hart->f[d] = as_u64(fmax(a, b));
        return false;
    case 0x61: { // FCVT.W.D / FCVT.WU.D / FCVT.L.D / FCVT.LU.D
        uint32_t s2 = rs2(insn);
        double r = roundd_rm(a, erm);
        switch (s2) {
        case 0: { // FCVT.W.D
            int64_t v;
            if (isnan(a) || r >= 0x1.0p31) { v = INT32_MAX; hart->fcsr |= FFLAG_NV; }
            else if (r < -0x1.0p31) { v = (int64_t)(int32_t)INT32_MIN; hart->fcsr |= FFLAG_NV; }
            else v = (int64_t)(int32_t)r;
            if (d) hart->x[d] = v;
            return false;
        }
        case 1: { // FCVT.WU.D
            int64_t v;
            if (isnan(a) || r >= 0x1.0p32) { v = (int64_t)(int32_t)UINT32_MAX; hart->fcsr |= FFLAG_NV; }
            else if (r < 0.0) { v = 0; if (r != 0.0) hart->fcsr |= FFLAG_NV; }
            else v = (int64_t)(int32_t)(uint32_t)r;
            if (d) hart->x[d] = v;
            return false;
        }
        case 2: { // FCVT.L.D
            int64_t v;
            if (isnan(a) || r >= 0x1.0p63) { v = INT64_MAX; hart->fcsr |= FFLAG_NV; }
            else if (r < -0x1.0p63) { v = INT64_MIN; hart->fcsr |= FFLAG_NV; }
            else v = (int64_t)r;
            if (d) hart->x[d] = (uint64_t)v;
            return false;
        }
        case 3: { // FCVT.LU.D
            uint64_t v;
            if (isnan(a) || r >= 0x1.0p64) { v = UINT64_MAX; hart->fcsr |= FFLAG_NV; }
            else if (r < 0.0) { v = 0; if (r != 0.0) hart->fcsr |= FFLAG_NV; }
            else v = (uint64_t)r;
            if (d) hart->x[d] = v;
            return false;
        }
        }
        break;
    }
    case 0x69: { // FCVT.D.W / FCVT.D.WU / FCVT.D.L / FCVT.D.LU
        uint32_t s2 = rs2(insn);
        // int32/uint32 → double is always exact; int64/uint64 → double may round
        switch (s2) {
        case 0: hart->f[d] = as_u64((double)(int32_t)hart->x[rs1(insn)]); return false;
        case 1: hart->f[d] = as_u64((double)(uint32_t)hart->x[rs1(insn)]); return false;
        case 2: hart->f[d] = as_u64(__ll2double_rn((long long)hart->x[rs1(insn)])); return false;
        case 3: hart->f[d] = as_u64(__ull2double_rn((unsigned long long)hart->x[rs1(insn)])); return false;
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
        case 2: // FEQ.D — NV only on signaling NaN
            if (is_snan_d(as_u64(a)) || is_snan_d(as_u64(b))) hart->fcsr |= FFLAG_NV;
            if (d) hart->x[d] = (a == b) ? 1 : 0; return false;
        case 1: // FLT.D — NV on any NaN
            if (isnan(a) || isnan(b)) hart->fcsr |= FFLAG_NV;
            if (d) hart->x[d] = (a < b) ? 1 : 0; return false;
        case 0: // FLE.D — NV on any NaN
            if (isnan(a) || isnan(b)) hart->fcsr |= FFLAG_NV;
            if (d) hart->x[d] = (a <= b) ? 1 : 0; return false;
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
