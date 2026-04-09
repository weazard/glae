#pragma once
#include "../core/hart.cuh"

// ============================================================
// Trap entry
// ============================================================
__device__ void take_trap(HartState* hart, uint64_t cause, uint64_t pc, uint64_t tval) {
    bool is_interrupt = (cause >> 63) & 1;
    uint64_t code = cause & ~CAUSE_INTERRUPT_BIT;


    // Determine target privilege: check delegation
    bool delegate = false;
    if (hart->priv <= PRV_S) {
        if (is_interrupt)
            delegate = (hart->mideleg >> code) & 1;
        else
            delegate = (hart->medeleg >> code) & 1;
    }

    if (delegate) {
        // Trap to S-mode
        hart->sepc = pc;
        hart->scause = cause;
        hart->stval = tval;
        hart->set_spp(hart->priv);

        // SPIE = SIE, SIE = 0
        if (hart->mstatus & MSTATUS_SIE) hart->mstatus |= MSTATUS_SPIE;
        else                              hart->mstatus &= ~MSTATUS_SPIE;
        hart->mstatus &= ~MSTATUS_SIE;

        // PC = stvec
        uint64_t base = hart->stvec & ~3ULL;
        uint64_t mode = hart->stvec & 3;
        if (mode == 1 && is_interrupt)
            hart->pc = base + 4 * code;
        else
            hart->pc = base;

        hart->priv = PRV_S;
    } else {
        // Trap to M-mode
        hart->mepc = pc;
        hart->mcause = cause;
        hart->mtval = tval;
        hart->set_mpp(hart->priv);

        if (hart->mstatus & MSTATUS_MIE) hart->mstatus |= MSTATUS_MPIE;
        else                              hart->mstatus &= ~MSTATUS_MPIE;
        hart->mstatus &= ~MSTATUS_MIE;

        uint64_t base = hart->mtvec & ~3ULL;
        uint64_t mode = hart->mtvec & 3;
        if (mode == 1 && is_interrupt)
            hart->pc = base + 4 * code;
        else
            hart->pc = base;

        hart->priv = PRV_M;
    }
}

// ============================================================
// MRET
// ============================================================
__device__ void do_mret(HartState* hart) {
    // Restore privilege
    hart->priv = hart->mpp();

    // MIE = MPIE, MPIE = 1, MPP = U (or lowest supported)
    if (hart->mstatus & MSTATUS_MPIE) hart->mstatus |= MSTATUS_MIE;
    else                               hart->mstatus &= ~MSTATUS_MIE;
    hart->mstatus |= MSTATUS_MPIE;
    hart->set_mpp(PRV_U);

    // Clear MPRV if MPP != M
    if (hart->priv != PRV_M)
        hart->mstatus &= ~MSTATUS_MPRV;

    hart->pc = hart->mepc;
}

// ============================================================
// SRET
// ============================================================
__device__ void do_sret(HartState* hart) {
    hart->priv = hart->spp();

    if (hart->mstatus & MSTATUS_SPIE) hart->mstatus |= MSTATUS_SIE;
    else                               hart->mstatus &= ~MSTATUS_SIE;
    hart->mstatus |= MSTATUS_SPIE;
    hart->set_spp(PRV_U);

    // Clear MPRV if SPP != M (always true since SPP is U or S)
    hart->mstatus &= ~MSTATUS_MPRV;

    hart->pc = hart->sepc;
}

// ============================================================
// Check and take pending interrupts
// ============================================================
__device__ bool check_interrupts(HartState* hart) {
    uint64_t pending = hart->mip & hart->mie;
    if (pending == 0) return false;

    // Check which interrupts can fire based on privilege and enable bits
    // M-mode interrupts: fire if priv < M, or (priv == M && MIE)
    // S-mode interrupts: fire if priv < S, or (priv == S && SIE)
    uint64_t m_enabled = 0, s_enabled = 0;

    if (hart->priv < PRV_M || (hart->priv == PRV_M && (hart->mstatus & MSTATUS_MIE)))
        m_enabled = ~hart->mideleg; // non-delegated interrupts go to M-mode

    if (hart->priv < PRV_S || (hart->priv == PRV_S && (hart->mstatus & MSTATUS_SIE)))
        s_enabled = hart->mideleg;  // delegated interrupts go to S-mode

    uint64_t actionable = pending & (m_enabled | s_enabled);
    if (actionable == 0) return false;

    // Priority: MEI > MSI > MTI > SEI > SSI > STI
    static const int prio[] = { INT_M_EXTERNAL, INT_M_SOFTWARE, INT_M_TIMER,
                                INT_S_EXTERNAL, INT_S_SOFTWARE, INT_S_TIMER };
    for (int i = 0; i < 6; i++) {
        if (actionable & (1ULL << prio[i])) {
            take_trap(hart, CAUSE_INTERRUPT_BIT | prio[i], hart->pc, 0);
            return true;
        }
    }
    return false;
}
