#pragma once
#include "../core/hart.cuh"

// Forward declaration
__device__ void take_trap(HartState* hart, uint64_t cause, uint64_t pc, uint64_t tval);

// ============================================================
// CSR read
// ============================================================
__device__ bool csr_read(HartState* hart, uint32_t addr, uint64_t* val) {
    // Privilege check: CSR[9:8] = minimum privilege
    uint32_t min_priv = (addr >> 8) & 3;
    if (hart->priv < min_priv) return false;

    switch (addr) {
    // --- FP ---
    case CSR_FFLAGS:    *val = hart->fcsr & 0x1F; return true;
    case CSR_FRM:       *val = (hart->fcsr >> 5) & 7; return true;
    case CSR_FCSR:      *val = hart->fcsr & 0xFF; return true;

    // --- Supervisor ---
    case CSR_SSTATUS:   *val = hart->mstatus & SSTATUS_MASK; return true;
    case CSR_SIE:       *val = hart->mie & SIP_MASK; return true;
    case CSR_STVEC:     *val = hart->stvec; return true;
    case CSR_SCOUNTEREN: *val = hart->scounteren; return true;
    case CSR_SENVCFG:   *val = hart->senvcfg; return true;
    case CSR_SSCRATCH:  *val = hart->sscratch; return true;
    case CSR_SEPC:      *val = hart->sepc; return true;
    case CSR_SCAUSE:    *val = hart->scause; return true;
    case CSR_STVAL:     *val = hart->stval; return true;
    case CSR_SIP:       *val = hart->mip & SIP_MASK; return true;
    case CSR_SATP:      *val = hart->satp; return true;
    case CSR_STIMECMP:  *val = hart->stimecmp; return true;

    // --- Machine ---
    case CSR_MSTATUS:   *val = hart->mstatus; return true;
    case CSR_MISA:      *val = hart->misa; return true;
    case CSR_MEDELEG:   *val = hart->medeleg; return true;
    case CSR_MIDELEG:   *val = hart->mideleg; return true;
    case CSR_MIE:       *val = hart->mie; return true;
    case CSR_MTVEC:     *val = hart->mtvec; return true;
    case CSR_MCOUNTEREN: *val = hart->mcounteren; return true;
    case CSR_MENVCFG:   *val = hart->menvcfg; return true;
    case CSR_MSCRATCH:  *val = hart->mscratch; return true;
    case CSR_MEPC:      *val = hart->mepc; return true;
    case CSR_MCAUSE:    *val = hart->mcause; return true;
    case CSR_MTVAL:     *val = hart->mtval; return true;
    case CSR_MIP:       *val = hart->mip; return true;

    // --- Machine info ---
    case CSR_MVENDORID: *val = 0; return true;
    case CSR_MARCHID:   *val = 0; return true;
    case CSR_MIMPID:    *val = 0; return true;
    case CSR_MHARTID:   *val = hart->mhartid; return true;

    // --- Counters ---
    case CSR_MCYCLE:
    case CSR_CYCLE:     *val = hart->instret; return true;  // cycle ≈ instret
    case CSR_MINSTRET:
    case CSR_INSTRET:   *val = hart->instret; return true;
    case CSR_TIME:      *val = hart->get_mtime(); return true;

    // --- PMP (stub — return 0) ---
    case CSR_PMPCFG0:
    case CSR_PMPCFG2:   *val = 0; return true;
    default:
        if (addr >= CSR_PMPADDR0 && addr < CSR_PMPADDR0 + 16) {
            *val = 0; return true;
        }
        // Unknown CSR
        return false;
    }
}

// ============================================================
// CSR write
// ============================================================
__device__ bool csr_write(HartState* hart, uint32_t addr, uint64_t val) {
    uint32_t min_priv = (addr >> 8) & 3;
    if (hart->priv < min_priv) return false;

    // Read-only CSRs: bits[11:10] == 0b11
    if ((addr >> 10 & 3) == 3) return false;

    switch (addr) {
    // --- FP ---
    case CSR_FFLAGS:
        hart->fcsr = (hart->fcsr & ~0x1F) | (val & 0x1F);
        hart->mark_fs_dirty();
        return true;
    case CSR_FRM:
        hart->fcsr = (hart->fcsr & ~0xE0) | ((val & 7) << 5);
        hart->mark_fs_dirty();
        return true;
    case CSR_FCSR:
        hart->fcsr = val & 0xFF;
        hart->mark_fs_dirty();
        return true;

    // --- Supervisor ---
    case CSR_SSTATUS: {
        uint64_t mask = SSTATUS_MASK & ~MSTATUS_SD; // SD is read-only
        hart->mstatus = (hart->mstatus & ~mask) | (val & mask);
        // Update SD based on FS/XS
        uint64_t fs = (hart->mstatus >> 13) & 3;
        uint64_t xs = (hart->mstatus >> 15) & 3;
        if (fs == 3 || xs == 3) hart->mstatus |= MSTATUS_SD;
        else hart->mstatus &= ~MSTATUS_SD;
        return true;
    }
    case CSR_SIE:
        hart->mie = (hart->mie & ~SIP_MASK) | (val & SIP_MASK);
        return true;
    case CSR_STVEC:
        hart->stvec = val & ~2ULL; // mode must be 0 or 1, not 2-3
        return true;
    case CSR_SCOUNTEREN:
        hart->scounteren = val & 7; // CY, TM, IR
        return true;
    case CSR_SENVCFG:
        hart->senvcfg = val;
        return true;
    case CSR_SSCRATCH:
        hart->sscratch = val;
        return true;
    case CSR_SEPC:
        hart->sepc = val & ~1ULL; // bit 0 always 0 (C ext: bit 0 = 0)
        return true;
    case CSR_SCAUSE:
        hart->scause = val;
        return true;
    case CSR_STVAL:
        hart->stval = val;
        return true;
    case CSR_SIP:
        // Only SSIP is writable from S-mode
        hart->mip = (hart->mip & ~MIP_SSIP) | (val & MIP_SSIP);
        return true;
    case CSR_SATP:
        // TVM: if set in mstatus, satp write traps from S-mode
        if (hart->priv == PRV_S && (hart->mstatus & MSTATUS_TVM))
            return false;
        hart->satp = val;
        // Flush TLB on satp write
        tlb_flush(hart->itlb);
        tlb_flush(hart->dtlb);
        return true;
    case CSR_STIMECMP:
        hart->stimecmp = val;
        // Clear timer pending
        hart->mip &= ~MIP_STIP;
        return true;

    // --- Machine ---
    case CSR_MSTATUS: {
        // WARL: only writable bits
        uint64_t mask = MSTATUS_SIE | MSTATUS_MIE | MSTATUS_SPIE | MSTATUS_MPIE |
                        MSTATUS_SPP | MSTATUS_MPP_MASK | MSTATUS_FS_MASK |
                        MSTATUS_MPRV | MSTATUS_SUM | MSTATUS_MXR |
                        MSTATUS_TVM | MSTATUS_TW | MSTATUS_TSR | MSTATUS_VS_MASK;
        hart->mstatus = (hart->mstatus & ~mask) | (val & mask);
        // Force UXL = 2 (RV64), SXL = 2
        hart->mstatus = (hart->mstatus & ~MSTATUS_UXL_MASK) | (2ULL << 32);
        hart->mstatus = (hart->mstatus & ~MSTATUS_SXL_MASK) | (2ULL << 34);
        // Update SD
        uint64_t fs = (hart->mstatus >> 13) & 3;
        uint64_t vs = (hart->mstatus >> 9) & 3;
        uint64_t xs = (hart->mstatus >> 15) & 3;
        if (fs == 3 || xs == 3 || vs == 3) hart->mstatus |= MSTATUS_SD;
        else hart->mstatus &= ~MSTATUS_SD;
        return true;
    }
    case CSR_MISA:
        return true; // WARL: ignore writes (fixed ISA)
    case CSR_MEDELEG:
        hart->medeleg = val & 0xB3FF; // delegatable exceptions
        return true;
    case CSR_MIDELEG:
        hart->mideleg = val & 0x222; // SSI, STI, SEI
        return true;
    case CSR_MIE:
        hart->mie = val;
        return true;
    case CSR_MTVEC:
        hart->mtvec = val & ~2ULL;
        return true;
    case CSR_MCOUNTEREN:
        hart->mcounteren = val & 7;
        return true;
    case CSR_MENVCFG:
        hart->menvcfg = val;
        return true;
    case CSR_MSCRATCH:
        hart->mscratch = val;
        return true;
    case CSR_MEPC:
        hart->mepc = val & ~1ULL;
        return true;
    case CSR_MCAUSE:
        hart->mcause = val;
        return true;
    case CSR_MTVAL:
        hart->mtval = val;
        return true;
    case CSR_MIP: {
        // Only MSIP, SSIP, STIP are writable
        uint64_t wmask = MIP_SSIP | MIP_STIP | MIP_MSIP;
        hart->mip = (hart->mip & ~wmask) | (val & wmask);
        return true;
    }
    // PMP stubs
    case CSR_PMPCFG0:
    case CSR_PMPCFG2:
        return true;
    default:
        if (addr >= CSR_PMPADDR0 && addr < CSR_PMPADDR0 + 16)
            return true;
        return false;
    }
}

// CSRRW/CSRRS/CSRRC execution helper
// Returns true if PC was changed (trap on illegal CSR)
__device__ bool exec_csr(HartState* hart, uint32_t insn) {
    uint32_t f3 = funct3(insn);
    uint32_t d = rd(insn);
    uint32_t s = rs1(insn);
    uint32_t csr_addr = insn >> 20;

    uint64_t old_val = 0;
    uint64_t write_val;

    bool do_write = true;

    // For CSRRS/CSRRC with rs1=0, no write
    if ((f3 == 2 || f3 == 3 || f3 == 6 || f3 == 7) && s == 0)
        do_write = false;
    // For CSRRW with rd=0, no read needed (but we still read for simplicity)

    if (!csr_read(hart, csr_addr, &old_val)) {
        take_trap(hart, EXC_ILLEGAL_INSN, hart->pc, insn);
        return true;
    }

    switch (f3) {
    case 1: // CSRRW
        write_val = hart->x[s];
        break;
    case 2: // CSRRS
        write_val = old_val | hart->x[s];
        break;
    case 3: // CSRRC
        write_val = old_val & ~hart->x[s];
        break;
    case 5: // CSRRWI
        write_val = (uint64_t)s; // zimm
        break;
    case 6: // CSRRSI
        write_val = old_val | (uint64_t)s;
        break;
    case 7: // CSRRCI
        write_val = old_val & ~(uint64_t)s;
        break;
    default:
        take_trap(hart, EXC_ILLEGAL_INSN, hart->pc, insn);
        return true;
    }

    if (do_write) {
        if (!csr_write(hart, csr_addr, write_val)) {
            take_trap(hart, EXC_ILLEGAL_INSN, hart->pc, insn);
            return true;
        }
    }

    if (d != 0) hart->x[d] = old_val;
    return false;
}
