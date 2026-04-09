#pragma once
#include <cstdint>

// ============================================================
// Privilege modes
// ============================================================
enum PrivMode : uint8_t { PRV_U = 0, PRV_S = 1, PRV_M = 3 };

// ============================================================
// SBI HSM hart states
// ============================================================
enum HartStatus : uint32_t {
    HSM_STARTED        = 0,
    HSM_STOPPED        = 1,
    HSM_START_PENDING   = 2,
    HSM_STOP_PENDING    = 3,
    HSM_SUSPENDED       = 4,
};

// Maximum harts supported
#define MAX_HARTS 256

// ============================================================
// Yield reasons (GPU → CPU communication)
// ============================================================
enum YieldReason : uint32_t {
    YIELD_NONE      = 0,
    YIELD_UART_TX   = 1,
    YIELD_WFI       = 2,
    YIELD_HALT      = 3,
    YIELD_BATCH_END = 4,
    YIELD_FATAL     = 5,
};

// ============================================================
// Exception cause codes (mcause/scause, interrupt bit = 0)
// ============================================================
#define EXC_INSN_MISALIGNED     0
#define EXC_INSN_ACCESS_FAULT   1
#define EXC_ILLEGAL_INSN        2
#define EXC_BREAKPOINT          3
#define EXC_LOAD_MISALIGNED     4
#define EXC_LOAD_ACCESS_FAULT   5
#define EXC_STORE_MISALIGNED    6
#define EXC_STORE_ACCESS_FAULT  7
#define EXC_ECALL_U             8
#define EXC_ECALL_S             9
#define EXC_ECALL_M             11
#define EXC_INSN_PAGE_FAULT     12
#define EXC_LOAD_PAGE_FAULT     13
#define EXC_STORE_PAGE_FAULT    15

// Interrupt cause codes (mcause/scause high bit set + code)
#define INT_S_SOFTWARE  1
#define INT_M_SOFTWARE  3
#define INT_S_TIMER     5
#define INT_M_TIMER     7
#define INT_S_EXTERNAL  9
#define INT_M_EXTERNAL  11

#define CAUSE_INTERRUPT_BIT (1ULL << 63)

// ============================================================
// mstatus / sstatus bit masks
// ============================================================
#define MSTATUS_SIE     (1ULL << 1)
#define MSTATUS_MIE     (1ULL << 3)
#define MSTATUS_SPIE    (1ULL << 5)
#define MSTATUS_UBE     (1ULL << 6)
#define MSTATUS_MPIE    (1ULL << 7)
#define MSTATUS_SPP     (1ULL << 8)
#define MSTATUS_VS_MASK (3ULL << 9)
#define MSTATUS_MPP_MASK (3ULL << 11)
#define MSTATUS_FS_MASK (3ULL << 13)
#define MSTATUS_XS_MASK (3ULL << 15)
#define MSTATUS_MPRV    (1ULL << 17)
#define MSTATUS_SUM     (1ULL << 18)
#define MSTATUS_MXR     (1ULL << 19)
#define MSTATUS_TVM     (1ULL << 20)
#define MSTATUS_TW      (1ULL << 21)
#define MSTATUS_TSR     (1ULL << 22)
#define MSTATUS_UXL_MASK (3ULL << 32)
#define MSTATUS_SXL_MASK (3ULL << 34)
#define MSTATUS_SD      (1ULL << 63)

#define MSTATUS_MPP_SHIFT 11
#define MSTATUS_FS_SHIFT  13

// sstatus visible bits
#define SSTATUS_MASK (MSTATUS_SIE | MSTATUS_SPIE | MSTATUS_UBE | MSTATUS_SPP | \
                      MSTATUS_FS_MASK | MSTATUS_XS_MASK | MSTATUS_VS_MASK | \
                      MSTATUS_SUM | MSTATUS_MXR | MSTATUS_UXL_MASK | MSTATUS_SD)

// ============================================================
// misa bits
// ============================================================
#define MISA_MXL_64 (2ULL << 62)
#define MISA_EXT_A  (1ULL << 0)
#define MISA_EXT_C  (1ULL << 2)
#define MISA_EXT_D  (1ULL << 3)
#define MISA_EXT_F  (1ULL << 5)
#define MISA_EXT_I  (1ULL << 8)
#define MISA_EXT_M  (1ULL << 12)
#define MISA_EXT_S  (1ULL << 18)
#define MISA_EXT_U  (1ULL << 20)

#define GLAE_MISA (MISA_MXL_64 | MISA_EXT_I | MISA_EXT_M | MISA_EXT_A | \
                   MISA_EXT_F | MISA_EXT_D | MISA_EXT_C | MISA_EXT_S | MISA_EXT_U)

// ============================================================
// CSR addresses
// ============================================================
// Supervisor
#define CSR_SSTATUS     0x100
#define CSR_SIE         0x104
#define CSR_STVEC       0x105
#define CSR_SCOUNTEREN  0x106
#define CSR_SENVCFG     0x10A
#define CSR_SSCRATCH    0x140
#define CSR_SEPC        0x141
#define CSR_SCAUSE      0x142
#define CSR_STVAL       0x143
#define CSR_SIP         0x144
#define CSR_STIMECMP    0x14D
#define CSR_SATP        0x180

// Machine
#define CSR_MSTATUS     0x300
#define CSR_MISA        0x301
#define CSR_MEDELEG     0x302
#define CSR_MIDELEG     0x303
#define CSR_MIE         0x304
#define CSR_MTVEC       0x305
#define CSR_MCOUNTEREN  0x306
#define CSR_MENVCFG     0x30A
#define CSR_MSCRATCH    0x340
#define CSR_MEPC        0x341
#define CSR_MCAUSE      0x342
#define CSR_MTVAL       0x343
#define CSR_MIP         0x344
#define CSR_PMPCFG0     0x3A0
#define CSR_PMPCFG2     0x3A2
#define CSR_PMPADDR0    0x3B0

// Counters
#define CSR_MCYCLE      0xB00
#define CSR_MINSTRET    0xB02
#define CSR_CYCLE       0xC00
#define CSR_TIME        0xC01
#define CSR_INSTRET     0xC02

// Machine info (read-only)
#define CSR_MVENDORID   0xF11
#define CSR_MARCHID     0xF12
#define CSR_MIMPID      0xF13
#define CSR_MHARTID     0xF14

// FP
#define CSR_FFLAGS      0x001
#define CSR_FRM         0x002
#define CSR_FCSR        0x003

// ============================================================
// Interrupt masks
// ============================================================
#define MIP_SSIP  (1ULL << INT_S_SOFTWARE)
#define MIP_MSIP  (1ULL << INT_M_SOFTWARE)
#define MIP_STIP  (1ULL << INT_S_TIMER)
#define MIP_MTIP  (1ULL << INT_M_TIMER)
#define MIP_SEIP  (1ULL << INT_S_EXTERNAL)
#define MIP_MEIP  (1ULL << INT_M_EXTERNAL)

// Bits visible as sip/sie
#define SIP_MASK  (MIP_SSIP | MIP_STIP | MIP_SEIP)

// ============================================================
// Memory map
// ============================================================
#define CLINT_BASE      0x02000000ULL
#define CLINT_SIZE      0x00010000ULL
#define PLIC_BASE       0x0C000000ULL
#define PLIC_SIZE       0x04000000ULL
#define UART0_BASE      0x10000000ULL
#define UART0_SIZE      0x00000100ULL
#define DRAM_BASE       0x80000000ULL
#define DRAM_SIZE_DEFAULT (128ULL * 1024 * 1024)  // 128 MB

// ============================================================
// Timer
// ============================================================
#define TIMEBASE_FREQ   10000000ULL  // 10 MHz

// ============================================================
// TLB
// ============================================================
#define TLB_SIZE 256

struct TLBEntry {
    uint64_t tag;     // vpn | (asid << 27)
    uint64_t ppn;
    uint8_t  perm;    // bits: [D:6][A:5][G:4][U:3][X:2][W:1][R:0]
    uint8_t  level;   // 0=4KB, 1=2MB, 2=1GB
    uint8_t  valid;
    uint8_t  pad;
};

enum AccessType : uint8_t {
    ACCESS_READ  = 0,
    ACCESS_WRITE = 1,
    ACCESS_EXEC  = 2,
};

// PTE permission bit positions
#define PTE_V (1ULL << 0)
#define PTE_R (1ULL << 1)
#define PTE_W (1ULL << 2)
#define PTE_X (1ULL << 3)
#define PTE_U (1ULL << 4)
#define PTE_G (1ULL << 5)
#define PTE_A (1ULL << 6)
#define PTE_D (1ULL << 7)

// ============================================================
// RV64GC opcodes
// ============================================================
#define OP_LOAD       0x03
#define OP_LOAD_FP    0x07
#define OP_MISC_MEM   0x0F
#define OP_OP_IMM     0x13
#define OP_AUIPC      0x17
#define OP_OP_IMM_32  0x1B
#define OP_STORE      0x23
#define OP_STORE_FP   0x27
#define OP_AMO        0x2F
#define OP_OP         0x33
#define OP_LUI        0x37
#define OP_OP_32      0x3B
#define OP_FMADD      0x43
#define OP_FMSUB      0x47
#define OP_FNMSUB     0x4B
#define OP_FNMADD     0x4F
#define OP_OP_FP      0x53
#define OP_BRANCH     0x63
#define OP_JALR       0x67
#define OP_JAL        0x6F
#define OP_SYSTEM     0x73

// ============================================================
// Instruction field extraction
// ============================================================
__device__ __forceinline__ uint32_t bits(uint32_t insn, int hi, int lo) {
    return (insn >> lo) & ((1U << (hi - lo + 1)) - 1);
}
__device__ __forceinline__ uint32_t rd(uint32_t insn)  { return bits(insn, 11, 7); }
__device__ __forceinline__ uint32_t rs1(uint32_t insn) { return bits(insn, 19, 15); }
__device__ __forceinline__ uint32_t rs2(uint32_t insn) { return bits(insn, 24, 20); }
__device__ __forceinline__ uint32_t rs3(uint32_t insn) { return bits(insn, 31, 27); }
__device__ __forceinline__ uint32_t funct3(uint32_t insn) { return bits(insn, 14, 12); }
__device__ __forceinline__ uint32_t funct7(uint32_t insn) { return bits(insn, 31, 25); }
__device__ __forceinline__ uint32_t funct5(uint32_t insn) { return bits(insn, 31, 27); }
__device__ __forceinline__ uint32_t opcode(uint32_t insn) { return insn & 0x7F; }

// Sign-extend from bit position
__device__ __forceinline__ int64_t sext(uint64_t val, int bit) {
    uint64_t mask = 1ULL << bit;
    return (int64_t)((val ^ mask) - mask);
}

// Immediate extraction
__device__ __forceinline__ int64_t imm_i(uint32_t insn) {
    return sext((uint64_t)(insn >> 20), 11);
}
__device__ __forceinline__ int64_t imm_s(uint32_t insn) {
    uint64_t v = ((insn >> 25) << 5) | bits(insn, 11, 7);
    return sext(v, 11);
}
__device__ __forceinline__ int64_t imm_b(uint32_t insn) {
    uint64_t v = (bits(insn, 31, 31) << 12) | (bits(insn, 7, 7) << 11) |
                 (bits(insn, 30, 25) << 5) | (bits(insn, 11, 8) << 1);
    return sext(v, 12);
}
__device__ __forceinline__ int64_t imm_u(uint32_t insn) {
    return sext((uint64_t)(insn & 0xFFFFF000), 31);
}
__device__ __forceinline__ int64_t imm_j(uint32_t insn) {
    uint64_t v = (bits(insn, 31, 31) << 20) | (bits(insn, 19, 12) << 12) |
                 (bits(insn, 20, 20) << 11) | (bits(insn, 30, 21) << 1);
    return sext(v, 20);
}

// ============================================================
// Hart state
// ============================================================
struct HartState {
    // Integer registers
    uint64_t x[32];

    // FP registers (raw bits — NaN-boxing managed by FP exec)
    uint64_t f[32];

    // Program counter
    uint64_t pc;

    // Privilege mode
    PrivMode priv;

    // Machine-level CSRs
    uint64_t mstatus;
    uint64_t misa;
    uint64_t medeleg;
    uint64_t mideleg;
    uint64_t mie;
    uint64_t mtvec;
    uint64_t mcounteren;
    uint64_t mscratch;
    uint64_t mepc;
    uint64_t mcause;
    uint64_t mtval;
    uint64_t mip;
    uint64_t menvcfg;

    // Supervisor-level CSRs (sstatus/sip/sie are views of m-level)
    uint64_t stvec;
    uint64_t scounteren;
    uint64_t sscratch;
    uint64_t sepc;
    uint64_t scause;
    uint64_t stval;
    uint64_t satp;
    uint64_t senvcfg;
    uint64_t stimecmp;

    // FP control/status
    uint32_t fcsr;  // fflags[4:0] | frm[7:5]

    // Read-only
    uint64_t mhartid;

    // Timer
    uint64_t gpu_clock_base;
    uint64_t gpu_clock_freq;

    // Counters
    uint64_t instret;

    // LR/SC reservation
    uint64_t reservation_addr;
    uint8_t  reservation_valid;

    // TLB
    TLBEntry itlb[TLB_SIZE];
    TLBEntry dtlb[TLB_SIZE];

    // Execution control
    uint32_t yield_reason;
    uint8_t  wfi;

    // SMP / HSM state
    HartStatus hsm_status;
    uint64_t   hsm_start_addr;   // PC to jump to when started
    uint64_t   hsm_start_arg;    // a1 value when started (opaque)

    // Set x[0] = 0 (call after every instruction)
    __device__ void zero_x0() { x[0] = 0; }

    // Get mtime from GPU clock
    __device__ uint64_t get_mtime() const {
        uint64_t elapsed = clock64() - gpu_clock_base;
        return (elapsed * TIMEBASE_FREQ) / gpu_clock_freq;
    }

    // FS field helpers
    __device__ uint32_t fs() const { return (mstatus >> MSTATUS_FS_SHIFT) & 3; }
    __device__ void set_fs(uint32_t v) {
        mstatus = (mstatus & ~MSTATUS_FS_MASK) | ((uint64_t)(v & 3) << MSTATUS_FS_SHIFT);
        if (v == 3) mstatus |= MSTATUS_SD;  // dirty → SD
    }
    __device__ void mark_fs_dirty() { if (fs() != 0) set_fs(3); }

    // MPP field helpers
    __device__ PrivMode mpp() const { return (PrivMode)((mstatus >> MSTATUS_MPP_SHIFT) & 3); }
    __device__ void set_mpp(PrivMode p) {
        mstatus = (mstatus & ~MSTATUS_MPP_MASK) | ((uint64_t)(p & 3) << MSTATUS_MPP_SHIFT);
    }
    __device__ PrivMode spp() const { return (mstatus & MSTATUS_SPP) ? PRV_S : PRV_U; }
    __device__ void set_spp(PrivMode p) {
        if (p == PRV_S) mstatus |= MSTATUS_SPP;
        else            mstatus &= ~MSTATUS_SPP;
    }
};
