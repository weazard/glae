#pragma once
#include "../core/hart.cuh"
#include "uart.cuh"
#include "clint.cuh"
#include "plic.cuh"

struct Machine {
    uint8_t*    dram;
    uint64_t    dram_size;
    Uart*       uart;
    Clint*      clint;
    Plic*       plic;
    HartState*  all_harts;   // array of all hart states (for IPI, CLINT)
    int         num_harts;
};

// Safe unaligned memory read/write
__device__ __forceinline__ uint64_t safe_read(const uint8_t* p, int size) {
    uint64_t v = 0;
    memcpy(&v, p, size);
    return v;
}

__device__ __forceinline__ void safe_write(uint8_t* p, uint64_t v, int size) {
    memcpy(p, &v, size);
}

// Physical memory load
__device__ bool bus_load(HartState* hart, Machine* m, uint64_t paddr, int size, uint64_t* val) {
    if (paddr >= DRAM_BASE && paddr < DRAM_BASE + m->dram_size) {
        uint64_t off = paddr - DRAM_BASE;
        *val = safe_read(m->dram + off, size);
        return true;
    }
    if (paddr >= UART0_BASE && paddr < UART0_BASE + UART0_SIZE) {
        *val = uart_read(m->uart, paddr - UART0_BASE);
        return true;
    }
    if (paddr >= CLINT_BASE && paddr < CLINT_BASE + CLINT_SIZE) {
        *val = clint_read(hart, m->clint, paddr - CLINT_BASE, size);
        return true;
    }
    if (paddr >= PLIC_BASE && paddr < PLIC_BASE + PLIC_SIZE) {
        *val = plic_read(m->plic, paddr - PLIC_BASE);
        return true;
    }
    *val = 0;
    return false;
}

// Physical memory store
__device__ bool bus_store(HartState* hart, Machine* m, uint64_t paddr, int size, uint64_t val) {
    if (paddr >= DRAM_BASE && paddr < DRAM_BASE + m->dram_size) {
        uint64_t off = paddr - DRAM_BASE;
        safe_write(m->dram + off, val, size);
        // Invalidate LR/SC reservation on store (all harts — SMP correctness)
        for (int h = 0; h < m->num_harts; h++) {
            if (m->all_harts[h].reservation_valid) {
                if (paddr <= m->all_harts[h].reservation_addr &&
                    m->all_harts[h].reservation_addr < paddr + size)
                    m->all_harts[h].reservation_valid = 0;
            }
        }
        return true;
    }
    if (paddr >= UART0_BASE && paddr < UART0_BASE + UART0_SIZE) {
        uart_write(hart, m->uart, paddr - UART0_BASE, val);
        return true;
    }
    if (paddr >= CLINT_BASE && paddr < CLINT_BASE + CLINT_SIZE) {
        clint_write(hart, m->all_harts, m->clint, paddr - CLINT_BASE, val, size);
        return true;
    }
    if (paddr >= PLIC_BASE && paddr < PLIC_BASE + PLIC_SIZE) {
        plic_write(m->plic, paddr - PLIC_BASE, val);
        return true;
    }
    return false;
}
