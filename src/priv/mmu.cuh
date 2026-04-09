#pragma once
#include "../core/hart.cuh"
#include "../devices/bus.cuh"

// ============================================================
// SV39 MMU — page table walk + software TLB
// ============================================================

// TLB operations
__device__ int tlb_index(uint64_t vpn) {
    return (int)((vpn ^ (vpn >> 8)) & (TLB_SIZE - 1));
}

__device__ bool tlb_lookup(TLBEntry* tlb, uint64_t vpn, uint16_t asid,
                           uint64_t* ppn, uint8_t* perm, uint8_t* level) {
    int idx = tlb_index(vpn);
    TLBEntry* e = &tlb[idx];
    if (!e->valid) return false;

    uint64_t tag = vpn | ((uint64_t)asid << 27);
    // Check exact match or global mapping
    bool global = (e->perm >> 4) & 1;
    uint64_t check_tag = global ? (e->tag & ((1ULL << 27) - 1)) : e->tag;
    uint64_t check_vpn = global ? vpn : tag;

    // For superpages, mask lower VPN bits
    uint64_t mask = 0;
    if (e->level == 1) mask = 0x1FF;        // 2MB: mask VPN[0]
    else if (e->level == 2) mask = 0x3FFFF; // 1GB: mask VPN[0:1]

    if ((check_vpn & ~mask) == (check_tag & ~mask)) {
        *ppn = e->ppn;
        *perm = e->perm;
        *level = e->level;
        return true;
    }
    return false;
}

__device__ void tlb_insert(TLBEntry* tlb, uint64_t vpn, uint16_t asid,
                           uint64_t ppn, uint8_t perm, uint8_t level) {
    int idx = tlb_index(vpn);
    TLBEntry* e = &tlb[idx];
    e->tag = vpn | ((uint64_t)asid << 27);
    e->ppn = ppn;
    e->perm = perm;
    e->level = level;
    e->valid = 1;
}

__device__ void tlb_flush(TLBEntry* tlb) {
    for (int i = 0; i < TLB_SIZE; i++) tlb[i].valid = 0;
}

__device__ void tlb_flush_addr(TLBEntry* tlb, uint64_t vpn) {
    int idx = tlb_index(vpn);
    tlb[idx].valid = 0;
}

// ============================================================
// SV39 Page Table Walk
// ============================================================
// Returns physical address, or takes a page fault trap and returns 0 with *fault=true
__device__ uint64_t sv39_walk(HartState* hart, Machine* m, uint64_t vaddr,
                              AccessType atype, bool* fault) {
    *fault = false;
    uint64_t satp = hart->satp;
    uint64_t ppn_base = satp & 0xFFFFFFFFFFFULL; // bits 43:0
    uint16_t asid = (satp >> 44) & 0xFFFF;

    uint64_t vpn[3];
    vpn[0] = (vaddr >> 12) & 0x1FF;
    vpn[1] = (vaddr >> 21) & 0x1FF;
    vpn[2] = (vaddr >> 30) & 0x1FF;

    // Check canonical address (bits 63:39 must equal bit 38)
    uint64_t top = vaddr >> 38;
    if (top != 0 && top != 0x3FFFFFF) {
        *fault = true;
        return 0;
    }

    uint64_t a = ppn_base * 4096;

    for (int i = 2; i >= 0; i--) {
        uint64_t pte_addr = a + vpn[i] * 8;

        // Read PTE from physical memory
        uint64_t pte = 0;
        if (pte_addr >= DRAM_BASE && pte_addr < DRAM_BASE + m->dram_size) {
            memcpy(&pte, m->dram + (pte_addr - DRAM_BASE), 8);
        } else {
            *fault = true;
            return 0;
        }

        // Check valid
        if (!(pte & PTE_V)) { *fault = true; return 0; }
        // R=0, W=1 is reserved
        if (!(pte & PTE_R) && (pte & PTE_W)) { *fault = true; return 0; }

        bool is_leaf = (pte & PTE_R) || (pte & PTE_X);

        if (!is_leaf) {
            // Pointer to next level
            if (i == 0) { *fault = true; return 0; } // no more levels
            a = ((pte >> 10) & 0xFFFFFFFFFFFULL) * 4096;
            continue;
        }

        // Leaf PTE found
        uint64_t pte_ppn = (pte >> 10) & 0xFFFFFFFFFFFULL;

        // Superpage alignment check
        if (i == 2 && (pte_ppn & 0x3FFFF)) { *fault = true; return 0; }
        if (i == 1 && (pte_ppn & 0x1FF))   { *fault = true; return 0; }

        // Permission checks
        bool user_page = (pte & PTE_U) != 0;
        PrivMode eff_priv = hart->priv;
        if ((hart->mstatus & MSTATUS_MPRV) && atype != ACCESS_EXEC)
            eff_priv = hart->mpp();

        if (eff_priv == PRV_U && !user_page) { *fault = true; return 0; }
        if (eff_priv == PRV_S && user_page && !(hart->mstatus & MSTATUS_SUM)) {
            *fault = true; return 0;
        }

        switch (atype) {
        case ACCESS_READ:
            if (!(pte & PTE_R) && !((hart->mstatus & MSTATUS_MXR) && (pte & PTE_X))) {
                *fault = true; return 0;
            }
            break;
        case ACCESS_WRITE:
            if (!(pte & PTE_W)) { *fault = true; return 0; }
            break;
        case ACCESS_EXEC:
            if (!(pte & PTE_X)) { *fault = true; return 0; }
            break;
        }

        // A/D bit management: set if not already set (Svadu behavior)
        bool need_update = false;
        if (!(pte & PTE_A)) { pte |= PTE_A; need_update = true; }
        if (atype == ACCESS_WRITE && !(pte & PTE_D)) { pte |= PTE_D; need_update = true; }

        if (need_update) {
            if (pte_addr >= DRAM_BASE && pte_addr < DRAM_BASE + m->dram_size) {
                memcpy(m->dram + (pte_addr - DRAM_BASE), &pte, 8);
            }
        }

        // Compute physical address
        uint64_t pa;
        uint64_t page_off = vaddr & 0xFFF;
        if (i == 2) {
            // 1GB gigapage
            pa = (pte_ppn << 12) | (vpn[1] << 21) | (vpn[0] << 12) | page_off;
        } else if (i == 1) {
            // 2MB megapage
            pa = (pte_ppn << 12) | (vpn[0] << 12) | page_off;
        } else {
            // 4KB page
            pa = (pte_ppn << 12) | page_off;
        }

        // Insert into TLB
        uint8_t perm = (uint8_t)(pte & 0xFF); // V|R|W|X|U|G|A|D
        uint64_t full_vpn = vaddr >> 12;
        TLBEntry* tlb = (atype == ACCESS_EXEC) ? hart->itlb : hart->dtlb;
        tlb_insert(tlb, full_vpn, asid, pte_ppn, perm, (uint8_t)i);

        return pa;
    }

    *fault = true;
    return 0;
}

// ============================================================
// Address translation (TLB + walk)
// ============================================================
// Forward declaration of take_trap (defined in trap.cuh)
__device__ void take_trap(HartState* hart, uint64_t cause, uint64_t pc, uint64_t tval);

__device__ bool translate(HartState* hart, Machine* m, uint64_t vaddr,
                          AccessType atype, uint64_t* paddr) {
    uint64_t mode = (hart->satp >> 60) & 0xF;

    // Bare mode or M-mode (without MPRV) → no translation
    PrivMode eff_priv = hart->priv;
    if ((hart->mstatus & MSTATUS_MPRV) && atype != ACCESS_EXEC)
        eff_priv = hart->mpp();

    if (mode == 0 || eff_priv == PRV_M) {
        *paddr = vaddr;
        return true;
    }

    // SV39 (mode == 8)
    if (mode != 8) {
        // Unknown mode, treat as fault
        uint64_t cause = (atype == ACCESS_EXEC) ? EXC_INSN_PAGE_FAULT :
                         (atype == ACCESS_READ)  ? EXC_LOAD_PAGE_FAULT :
                                                   EXC_STORE_PAGE_FAULT;
        take_trap(hart, cause, hart->pc, vaddr);
        return false;
    }

    // TLB lookup
    uint64_t vpn = vaddr >> 12;
    uint16_t asid = (hart->satp >> 44) & 0xFFFF;
    TLBEntry* tlb = (atype == ACCESS_EXEC) ? hart->itlb : hart->dtlb;
    uint64_t ppn;
    uint8_t perm, level;

    if (tlb_lookup(tlb, vpn, asid, &ppn, &perm, &level)) {
        // Quick permission check on TLB hit
        bool ok = true;
        PrivMode ep = hart->priv;
        if ((hart->mstatus & MSTATUS_MPRV) && atype != ACCESS_EXEC) ep = hart->mpp();

        bool user_page = (perm & (1 << 4)) != 0;
        if (ep == PRV_U && !user_page) ok = false;
        if (ep == PRV_S && user_page && !(hart->mstatus & MSTATUS_SUM)) ok = false;

        if (ok) {
            switch (atype) {
            case ACCESS_READ:
                ok = (perm & 0x02) || ((hart->mstatus & MSTATUS_MXR) && (perm & 0x08));
                break;
            case ACCESS_WRITE: ok = (perm & 0x04) != 0; break;
            case ACCESS_EXEC:  ok = (perm & 0x08) != 0; break;
            }
        }

        if (ok) {
            uint64_t page_off = vaddr & 0xFFF;
            if (level == 2) {
                *paddr = (ppn << 12) | ((vpn & 0x3FFFF) << 12) | page_off;
            } else if (level == 1) {
                *paddr = (ppn << 12) | ((vpn & 0x1FF) << 12) | page_off;
            } else {
                *paddr = (ppn << 12) | page_off;
            }
            return true;
        }
        // TLB hit but permission fail — fall through to walk for precise fault
    }

    // TLB miss or permission fail — full page table walk
    bool fault = false;
    uint64_t pa = sv39_walk(hart, m, vaddr, atype, &fault);
    if (fault) {
        uint64_t cause = (atype == ACCESS_EXEC) ? EXC_INSN_PAGE_FAULT :
                         (atype == ACCESS_READ)  ? EXC_LOAD_PAGE_FAULT :
                                                   EXC_STORE_PAGE_FAULT;
        take_trap(hart, cause, hart->pc, vaddr);
        return false;
    }

    *paddr = pa;
    return true;
}

// ============================================================
// Memory access wrappers (translate + bus access)
// ============================================================
__device__ bool mem_fetch(HartState* hart, Machine* m, uint32_t* insn) {
    uint64_t paddr;
    if (!translate(hart, m, hart->pc, ACCESS_EXEC, &paddr)) return false;

    // Fetch 4 bytes (may be compressed — caller checks bits[1:0])
    // Handle unaligned: compressed insn at end of page
    if (paddr >= DRAM_BASE && paddr < DRAM_BASE + m->dram_size) {
        uint64_t off = paddr - DRAM_BASE;
        if (off + 4 <= m->dram_size) {
            memcpy(insn, m->dram + off, 4);
        } else if (off + 2 <= m->dram_size) {
            *insn = 0;
            memcpy(insn, m->dram + off, 2);
        } else {
            take_trap(hart, EXC_INSN_ACCESS_FAULT, hart->pc, hart->pc);
            return false;
        }
        return true;
    }
    take_trap(hart, EXC_INSN_ACCESS_FAULT, hart->pc, hart->pc);
    return false;
}

__device__ bool mem_load(HartState* hart, Machine* m, uint64_t vaddr, int size, uint64_t* val) {
    // Alignment check
    if (vaddr & (size - 1)) {
        take_trap(hart, EXC_LOAD_MISALIGNED, hart->pc, vaddr);
        return false;
    }
    uint64_t paddr;
    if (!translate(hart, m, vaddr, ACCESS_READ, &paddr)) return false;
    if (!bus_load(hart, m, paddr, size, val)) {
        take_trap(hart, EXC_LOAD_ACCESS_FAULT, hart->pc, vaddr);
        return false;
    }
    return true;
}

__device__ bool mem_store(HartState* hart, Machine* m, uint64_t vaddr, int size, uint64_t val) {
    if (vaddr & (size - 1)) {
        take_trap(hart, EXC_STORE_MISALIGNED, hart->pc, vaddr);
        return false;
    }
    uint64_t paddr;
    if (!translate(hart, m, vaddr, ACCESS_WRITE, &paddr)) return false;
    if (!bus_store(hart, m, paddr, size, val)) {
        take_trap(hart, EXC_STORE_ACCESS_FAULT, hart->pc, vaddr);
        return false;
    }
    return true;
}

// Physical address translation for AMO (returns paddr, caller does bus ops)
__device__ bool mem_translate_store(HartState* hart, Machine* m, uint64_t vaddr,
                                    int size, uint64_t* paddr) {
    if (vaddr & (size - 1)) {
        take_trap(hart, EXC_STORE_MISALIGNED, hart->pc, vaddr);
        return false;
    }
    return translate(hart, m, vaddr, ACCESS_WRITE, paddr);
}
