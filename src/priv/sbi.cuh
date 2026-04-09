#pragma once
#include "../core/hart.cuh"
#include "../devices/uart.cuh"
#include "../devices/clint.cuh"
#include "../platform/debug.cuh"

// SBI extension IDs
#define SBI_EXT_BASE    0x10
#define SBI_EXT_TIMER   0x54494D45  // "TIME"
#define SBI_EXT_IPI     0x735049    // "sPI"
#define SBI_EXT_RFENCE  0x52464E43  // "RFNC"
#define SBI_EXT_HSM     0x48534D    // "HSM"
#define SBI_EXT_SRST    0x53525354  // "SRST"

// Legacy extensions
#define SBI_LEGACY_SET_TIMER       0
#define SBI_LEGACY_CONSOLE_PUTCHAR 1
#define SBI_LEGACY_CONSOLE_GETCHAR 2

// SBI error codes
#define SBI_SUCCESS              0
#define SBI_ERR_FAILED          -1
#define SBI_ERR_NOT_SUPPORTED   -2
#define SBI_ERR_INVALID_PARAM   -3

// Returns true = handled (PC should advance), false = not an SBI call
__device__ bool handle_sbi(HartState* hart, Machine* m) {
    uint64_t eid = hart->x[17]; // a7
    uint64_t fid = hart->x[16]; // a6
    uint64_t a0 = hart->x[10];

    DPRINTF("[SBI] eid=0x%llx fid=0x%llx a0=0x%llx\n",
            (unsigned long long)eid, (unsigned long long)fid, (unsigned long long)a0);

    int64_t err = SBI_SUCCESS;
    int64_t ret = 0;

    switch (eid) {
    // ---- Legacy extensions ----
    case SBI_LEGACY_SET_TIMER:
        m->clint->mtimecmp = a0;
        hart->mip &= ~MIP_STIP;
        // Legacy returns void — set a0 = 0
        hart->x[10] = 0;
        return true;

    case SBI_LEGACY_CONSOLE_PUTCHAR:
        ring_push(m->uart->tx_ring, (uint8_t)a0);
        hart->yield_reason = YIELD_UART_TX;
        hart->x[10] = 0;
        return true;

    case SBI_LEGACY_CONSOLE_GETCHAR: {
        uint8_t ch;
        if (ring_pop(m->uart->rx_ring, &ch))
            hart->x[10] = ch;
        else
            hart->x[10] = (uint64_t)-1;
        return true;
    }

    // ---- Base extension ----
    case SBI_EXT_BASE:
        switch (fid) {
        case 0: ret = 2; break;             // sbi_get_spec_version (v0.2)
        case 1: ret = 0x676C6165; break;    // sbi_get_impl_id ("glae")
        case 2: ret = 1; break;             // sbi_get_impl_version
        case 3: // sbi_probe_extension
            switch (a0) {
            case SBI_EXT_BASE:
            case SBI_EXT_TIMER:
            case SBI_EXT_RFENCE:
            case SBI_EXT_SRST:
            case SBI_LEGACY_SET_TIMER:
            case SBI_LEGACY_CONSOLE_PUTCHAR:
            case SBI_LEGACY_CONSOLE_GETCHAR:
                ret = 1; break;
            default:
                ret = 0; break;
            }
            break;
        case 4: ret = 0; break;  // mvendorid
        case 5: ret = 0; break;  // marchid
        case 6: ret = 0; break;  // mimpid
        default: err = SBI_ERR_NOT_SUPPORTED; break;
        }
        break;

    // ---- Timer extension ----
    case SBI_EXT_TIMER:
        if (fid == 0) {
            // sbi_set_timer
            hart->stimecmp = a0;
            m->clint->mtimecmp = a0;
            hart->mip &= ~MIP_STIP;
        } else {
            err = SBI_ERR_NOT_SUPPORTED;
        }
        break;

    // ---- IPI extension ----
    case SBI_EXT_IPI:
        if (fid == 0) {
            // sbi_send_ipi — single hart, just set SSIP
            hart->mip |= MIP_SSIP;
        } else {
            err = SBI_ERR_NOT_SUPPORTED;
        }
        break;

    // ---- RFENCE extension ----
    case SBI_EXT_RFENCE:
        // All fence functions: just flush TLB
        tlb_flush(hart->itlb);
        tlb_flush(hart->dtlb);
        break;

    // ---- HSM extension ----
    case SBI_EXT_HSM:
        if (fid == 0) {
            // sbi_hart_start — single hart, not supported
            err = SBI_ERR_NOT_SUPPORTED;
        } else if (fid == 1) {
            // sbi_hart_stop
            hart->yield_reason = YIELD_HALT;
        } else if (fid == 2) {
            // sbi_hart_get_status — STARTED
            ret = 0;
        } else {
            err = SBI_ERR_NOT_SUPPORTED;
        }
        break;

    // ---- SRST extension ----
    case SBI_EXT_SRST:
        if (fid == 0) {
            // sbi_system_reset
            hart->yield_reason = YIELD_HALT;
        } else {
            err = SBI_ERR_NOT_SUPPORTED;
        }
        break;

    default:
        // Unknown extension
        err = SBI_ERR_NOT_SUPPORTED;
        break;
    }

    // SBI v0.2+ return convention: a0 = error, a1 = value
    hart->x[10] = (uint64_t)err;
    hart->x[11] = (uint64_t)ret;
    return true;
}
