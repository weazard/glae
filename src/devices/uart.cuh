#pragma once
#include "../core/hart.cuh"
#include "../platform/ring.h"

// 16550A UART registers
struct Uart {
    uint8_t  ier;       // interrupt enable
    uint8_t  iir;       // interrupt identification (read-only)
    uint8_t  fcr;       // FIFO control
    uint8_t  lcr;       // line control
    uint8_t  mcr;       // modem control
    uint8_t  scr;       // scratch
    uint16_t divisor;   // baud divisor (DLL | DLM<<8)
    Ring*    tx_ring;   // pinned memory — GPU writes, CPU reads
    Ring*    rx_ring;   // pinned memory — CPU writes, GPU reads
    uint8_t  irq_pending; // for PLIC
};

__device__ uint8_t uart_lsr(Uart* u) {
    uint8_t lsr = 0x60;  // THR empty + transmitter idle (bits 5,6)
    if (!ring_empty(u->rx_ring))
        lsr |= 0x01;     // data ready (bit 0)
    return lsr;
}

// Compute actual IIR based on IER and pending conditions
__device__ uint8_t uart_compute_iir(Uart* u) {
    // Priority: Receiver Line Status (0x06) > RX Data (0x04) > THR Empty (0x02)
    if ((u->ier & 0x01) && !ring_empty(u->rx_ring))
        return 0x04;  // Received Data Available, interrupt pending (bit 0 = 0)
    if ((u->ier & 0x02) && u->irq_pending)
        return 0x02;  // THR Empty (only after TX write re-arms it)
    return 0x01;       // No interrupt pending (bit 0 = 1)
}

__device__ uint64_t uart_read(Uart* u, uint64_t offset) {
    bool dlab = (u->lcr >> 7) & 1;
    switch (offset & 0x7) {
    case 0:
        if (dlab) return u->divisor & 0xFF;
        else {
            uint8_t ch = 0;
            ring_pop(u->rx_ring, &ch);
            return ch;
        }
    case 1:
        if (dlab) return (u->divisor >> 8) & 0xFF;
        else return u->ier;
    case 2: {
        uint8_t iir = uart_compute_iir(u);
        if (u->fcr & 0x01) iir |= 0xC0;  // FIFOs enabled bits
        // Reading IIR clears THR Empty interrupt
        if ((iir & 0x0E) == 0x02) u->irq_pending = 0;
        return iir;
    }
    case 3: return u->lcr;
    case 4: return u->mcr;
    case 5: return uart_lsr(u);
    case 6: return 0;  // MSR: no modem signals
    case 7: return u->scr;
    default: return 0;
    }
}

__device__ void uart_write(HartState* hart, Uart* u, uint64_t offset, uint64_t val) {
    bool dlab = (u->lcr >> 7) & 1;
    uint8_t v = (uint8_t)val;
    switch (offset & 0x7) {
    case 0:
        if (dlab) { u->divisor = (u->divisor & 0xFF00) | v; }
        else {
            ring_push(u->tx_ring, v);
            u->irq_pending = 1;  // Re-arm THR Empty interrupt
            // Only yield when ring is getting full — batch characters
            if (ring_count(u->tx_ring) > RING_SIZE * 3 / 4) {
                if (hart->yield_reason == YIELD_NONE)
                    hart->yield_reason = YIELD_UART_TX;
            }
        }
        break;
    case 1:
        if (dlab) u->divisor = (u->divisor & 0x00FF) | ((uint16_t)v << 8);
        else u->ier = v & 0x0F;
        break;
    case 2: u->fcr = v; break;
    case 3: u->lcr = v; break;
    case 4: u->mcr = v & 0x1F; break;
    case 5: break;  // LSR read-only
    case 6: break;  // MSR read-only
    case 7: u->scr = v; break;
    }
}
