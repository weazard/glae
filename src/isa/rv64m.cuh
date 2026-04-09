#pragma once
#include "../core/hart.cuh"

// ============================================================
// M Extension — Multiply / Divide (64-bit)
// ============================================================
__device__ void exec_mul(HartState* hart, uint32_t insn) {
    uint32_t d = rd(insn);
    uint64_t a = hart->x[rs1(insn)];
    uint64_t b = hart->x[rs2(insn)];
    uint32_t f = funct3(insn);
    uint64_t result;

    switch (f) {
    case 0: // MUL
        result = a * b;
        break;
    case 1: { // MULH (signed × signed, high 64)
        uint64_t hi = __umul64hi(a, b);
        if ((int64_t)a < 0) hi -= b;
        if ((int64_t)b < 0) hi -= a;
        result = hi;
        break;
    }
    case 2: { // MULHSU (signed × unsigned, high 64)
        uint64_t hi = __umul64hi(a, b);
        if ((int64_t)a < 0) hi -= b;
        result = hi;
        break;
    }
    case 3: // MULHU (unsigned × unsigned, high 64)
        result = __umul64hi(a, b);
        break;
    case 4: // DIV
        if (b == 0)
            result = (uint64_t)-1;
        else if ((int64_t)a == INT64_MIN && (int64_t)b == -1)
            result = (uint64_t)INT64_MIN; // overflow
        else
            result = (uint64_t)((int64_t)a / (int64_t)b);
        break;
    case 5: // DIVU
        result = (b == 0) ? UINT64_MAX : a / b;
        break;
    case 6: // REM
        if (b == 0)
            result = a;
        else if ((int64_t)a == INT64_MIN && (int64_t)b == -1)
            result = 0;
        else
            result = (uint64_t)((int64_t)a % (int64_t)b);
        break;
    case 7: // REMU
        result = (b == 0) ? a : a % b;
        break;
    default: result = 0;
    }

    if (d != 0) hart->x[d] = result;
}

// ============================================================
// M Extension — 32-bit variants (MULW, DIVW, etc.)
// ============================================================
__device__ void exec_mul_32(HartState* hart, uint32_t insn) {
    uint32_t d = rd(insn);
    int32_t a = (int32_t)hart->x[rs1(insn)];
    int32_t b = (int32_t)hart->x[rs2(insn)];
    uint32_t f = funct3(insn);
    int32_t result;

    switch (f) {
    case 0: // MULW
        result = a * b;
        break;
    case 4: // DIVW
        if (b == 0)
            result = -1;
        else if (a == INT32_MIN && b == -1)
            result = INT32_MIN;
        else
            result = a / b;
        break;
    case 5: // DIVUW
        if ((uint32_t)b == 0)
            result = (int32_t)UINT32_MAX;
        else
            result = (int32_t)((uint32_t)a / (uint32_t)b);
        break;
    case 6: // REMW
        if (b == 0)
            result = a;
        else if (a == INT32_MIN && b == -1)
            result = 0;
        else
            result = a % b;
        break;
    case 7: // REMUW
        if ((uint32_t)b == 0)
            result = a;
        else
            result = (int32_t)((uint32_t)a % (uint32_t)b);
        break;
    default: result = 0;
    }

    if (d != 0) hart->x[d] = (int64_t)result; // sign-extend 32→64
}
