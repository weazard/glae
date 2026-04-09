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
#define SBI_ERR_ALREADY_AVAILABLE -6

// Returns true = handled (PC should advance)
__device__ bool handle_sbi(HartState* hart, Machine* m) {
    uint64_t eid = hart->x[17]; // a7
    uint64_t fid = hart->x[16]; // a6
    uint64_t a0 = hart->x[10];

    if (eid != SBI_EXT_TIMER && eid != SBI_LEGACY_SET_TIMER)
        DPRINTF("[SBI] hart%llu eid=0x%llx fid=0x%llx a0=0x%llx\n",
                (unsigned long long)hart->mhartid,
                (unsigned long long)eid, (unsigned long long)fid,
                (unsigned long long)a0);

    int64_t err = SBI_SUCCESS;
    int64_t ret = 0;
    uint32_t my_id = (uint32_t)hart->mhartid;

    switch (eid) {
    // ---- Legacy extensions ----
    case SBI_LEGACY_SET_TIMER:
        if (my_id < MAX_HARTS) {
            m->clint->mtimecmp[my_id] = a0;
            hart->mip &= ~MIP_STIP;
        }
        hart->x[10] = 0;
        return true;

    case SBI_LEGACY_CONSOLE_PUTCHAR:
        ring_push(m->uart->tx_ring, (uint8_t)a0);
        if (ring_count(m->uart->tx_ring) > RING_SIZE * 3 / 4)
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
        case 0: ret = 2; break;             // spec version 0.2
        case 1: ret = 0x676C6165; break;    // impl id "glae"
        case 2: ret = 1; break;             // impl version
        case 3: // probe extension
            switch (a0) {
            case SBI_EXT_BASE: case SBI_EXT_TIMER: case SBI_EXT_IPI:
            case SBI_EXT_RFENCE: case SBI_EXT_HSM: case SBI_EXT_SRST:
            case SBI_LEGACY_SET_TIMER: case SBI_LEGACY_CONSOLE_PUTCHAR:
            case SBI_LEGACY_CONSOLE_GETCHAR:
                ret = 1; break;
            default: ret = 0; break;
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
            hart->stimecmp = a0;
            if (my_id < MAX_HARTS) m->clint->mtimecmp[my_id] = a0;
            hart->mip &= ~MIP_STIP;
        } else {
            err = SBI_ERR_NOT_SUPPORTED;
        }
        break;

    // ---- IPI extension ----
    case SBI_EXT_IPI:
        if (fid == 0) {
            // sbi_send_ipi(hart_mask, hart_mask_base)
            uint64_t mask = a0;
            uint64_t base = hart->x[11]; // a1
            for (int i = 0; i < 64 && (base + i) < (uint64_t)m->num_harts; i++) {
                if (mask & (1ULL << i)) {
                    m->all_harts[base + i].mip |= MIP_SSIP;
                }
            }
        } else {
            err = SBI_ERR_NOT_SUPPORTED;
        }
        break;

    // ---- RFENCE extension ----
    case SBI_EXT_RFENCE:
        // All fence functions: flush this hart's TLB and icache
        tlb_flush(hart->itlb);
        tlb_flush(hart->dtlb);
        icache_flush();
        // For remote fences, also signal other harts (simplified: flush all)
        if (fid >= 1) {
            uint64_t mask = a0;
            uint64_t base = hart->x[11];
            for (int i = 0; i < 64 && (base + i) < (uint64_t)m->num_harts; i++) {
                if (mask & (1ULL << i)) {
                    HartState* target = &m->all_harts[base + i];
                    tlb_flush(target->itlb);
                    tlb_flush(target->dtlb);
                }
            }
        }
        break;

    // ---- HSM extension ----
    case SBI_EXT_HSM:
        switch (fid) {
        case 0: { // sbi_hart_start(hartid, start_addr, opaque)
            uint64_t target_id = a0;
            uint64_t start_addr = hart->x[11]; // a1
            uint64_t opaque = hart->x[12];     // a2
            if (target_id >= (uint64_t)m->num_harts) {
                err = SBI_ERR_INVALID_PARAM;
            } else {
                HartState* target = &m->all_harts[target_id];
                if (target->hsm_status == HSM_STOPPED) {
                    target->hsm_start_addr = start_addr;
                    target->hsm_start_arg = opaque;
                    target->hsm_status = HSM_START_PENDING;
                    DPRINTF("[HSM] hart%llu starting hart%llu at %llx\n",
                            (unsigned long long)my_id,
                            (unsigned long long)target_id,
                            (unsigned long long)start_addr);
                } else if (target->hsm_status == HSM_STARTED) {
                    err = SBI_ERR_ALREADY_AVAILABLE;
                } else {
                    err = SBI_ERR_FAILED;
                }
            }
            break;
        }
        case 1: // sbi_hart_stop
            hart->hsm_status = HSM_STOPPED;
            hart->yield_reason = YIELD_HALT;
            break;
        case 2: { // sbi_hart_get_status(hartid)
            uint64_t target_id = a0;
            if (target_id >= (uint64_t)m->num_harts)
                err = SBI_ERR_INVALID_PARAM;
            else
                ret = m->all_harts[target_id].hsm_status;
            break;
        }
        default:
            err = SBI_ERR_NOT_SUPPORTED;
            break;
        }
        break;

    // ---- SRST extension ----
    case SBI_EXT_SRST:
        if (fid == 0) {
            // Signal ALL harts to halt
            for (int i = 0; i < m->num_harts; i++)
                m->all_harts[i].yield_reason = YIELD_HALT;
        } else {
            err = SBI_ERR_NOT_SUPPORTED;
        }
        break;

    default:
        err = SBI_ERR_NOT_SUPPORTED;
        break;
    }

    hart->x[10] = (uint64_t)err;
    hart->x[11] = (uint64_t)ret;
    return true;
}
