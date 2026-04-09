// ============================================================
// GLAE — GPU-native RV64GC Hypervisor
// Phase 1: Serial interpreter
// ============================================================

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <unistd.h>
#include <fcntl.h>
#include <termios.h>

// Platform
#include "platform/ring.h"
#include "platform/debug.cuh"

// Core
#include "core/hart.cuh"

// ISA (order matters for dependencies)
#include "isa/rv64c.cuh"

// Devices
#include "devices/uart.cuh"
#include "devices/clint.cuh"
#include "devices/plic.cuh"
#include "devices/bus.cuh"

// Privilege
#include "priv/mmu.cuh"
#include "priv/csr.cuh"
#include "priv/trap.cuh"
#include "priv/sbi.cuh"

// ISA execution
#include "isa/rv64i.cuh"
#include "isa/rv64m.cuh"
#include "isa/rv64a.cuh"
#include "isa/rv64f.cuh"
#include "isa/rv64d.cuh"
#include "isa/system.cuh"

// Top-level decode
#include "core/decode.cuh"

// FDT
#include "platform/fdt.cuh"

// ============================================================
// GPU Kernel — main execution loop
// ============================================================
#define BATCH_SIZE 10000

__global__ void vcpu_run(HartState* hart, Machine* mach) {
    hart->yield_reason = YIELD_NONE;

    for (int i = 0; i < BATCH_SIZE && hart->yield_reason == YIELD_NONE; i++) {
        // Check timer periodically
        if ((i & 0xFF) == 0) {
            clint_tick(hart, mach->clint);
            // Check stimecmp too
            if (hart->stimecmp != 0 && hart->get_mtime() >= hart->stimecmp)
                hart->mip |= MIP_STIP;
            plic_update_ext(hart, mach->plic);
        }

        // Check interrupts
        if ((i & 0x3F) == 0) {
            if (hart->wfi) {
                uint64_t pending = hart->mip & hart->mie;
                if (pending) hart->wfi = 0;
                else { hart->yield_reason = YIELD_WFI; break; }
            }
            check_interrupts(hart);
        }

        // Fetch
        uint32_t raw = 0;
        if (!mem_fetch(hart, mach, &raw)) continue;

        // Decompress if needed
        uint32_t insn;
        int insn_len;
        if ((raw & 3) != 3) {
            insn = decompress_c((uint16_t)(raw & 0xFFFF));
            insn_len = 2;
            if (insn == 0) {
                take_trap(hart, EXC_ILLEGAL_INSN, hart->pc, raw & 0xFFFF);
                continue;
            }
        } else {
            insn = raw;
            insn_len = 4;
        }

        // Execute
        bool pc_written = execute(hart, mach, insn, insn_len);
        if (!pc_written)
            hart->pc += insn_len;

        hart->x[0] = 0;
        hart->instret++;
    }

    if (hart->yield_reason == YIELD_NONE)
        hart->yield_reason = YIELD_BATCH_END;
}

// ============================================================
// Host: setup and main loop
// ============================================================

static void init_hart(HartState* h, uint64_t entry_pc, uint64_t dtb_addr, uint64_t gpu_freq) {
    memset(h, 0, sizeof(HartState));

    h->pc = entry_pc;
    h->priv = PRV_S;

    // a0 = hartid, a1 = dtb address
    h->x[10] = 0;
    h->x[11] = dtb_addr;

    // misa: RV64IMAFDC + S + U
    h->misa = GLAE_MISA;

    // mstatus: FS=Initial(1), MPP=S, UXL=2, SXL=2
    h->mstatus = (1ULL << MSTATUS_FS_SHIFT) |  // FS = Initial
                 (2ULL << 32) |                  // UXL = RV64
                 (2ULL << 34);                   // SXL = RV64

    // Delegate standard traps and interrupts to S-mode
    // Exceptions: misaligned(0), illegal(2), breakpoint(3), ecall-U(8),
    //   insn page fault(12), load page fault(13), store page fault(15)
    h->medeleg = (1 << 0) | (1 << 1) | (1 << 2) | (1 << 3) |
                 (1 << 4) | (1 << 5) | (1 << 6) | (1 << 7) |
                 (1 << 8) | (1 << 12) | (1 << 13) | (1 << 15);
    // Interrupts: SSI(1), STI(5), SEI(9)
    h->mideleg = MIP_SSIP | MIP_STIP | MIP_SEIP;

    // Timer
    h->gpu_clock_freq = gpu_freq;
    h->stimecmp = UINT64_MAX;

    // TLB starts invalid
    for (int i = 0; i < TLB_SIZE; i++) {
        h->itlb[i].valid = 0;
        h->dtlb[i].valid = 0;
    }
}

static void set_nonblocking_stdin() {
    int flags = fcntl(STDIN_FILENO, F_GETFL, 0);
    fcntl(STDIN_FILENO, F_SETFL, flags | O_NONBLOCK);
}

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <kernel-image> [--dram-mb N]\n", argv[0]);
        return 1;
    }

    const char* kernel_path = argv[1];
    uint64_t dram_size = DRAM_SIZE_DEFAULT;

    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "--dram-mb") == 0 && i + 1 < argc) {
            dram_size = (uint64_t)atoi(argv[++i]) * 1024 * 1024;
        }
    }

    bool debug = getenv("GLAE_DEBUG") != nullptr;

    // Get GPU clock frequency
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    uint64_t gpu_freq = (uint64_t)prop.clockRate * 1000; // kHz → Hz

    if (debug) {
        printf("[GLAE] GPU: %s (%d SMs, %llu Hz clock)\n",
               prop.name, prop.multiProcessorCount, (unsigned long long)gpu_freq);
        printf("[GLAE] DRAM: %llu MB at 0x%llx\n",
               (unsigned long long)(dram_size / (1024*1024)),
               (unsigned long long)DRAM_BASE);
    }

    // Load kernel image
    FILE* f = fopen(kernel_path, "rb");
    if (!f) { perror("Failed to open kernel"); return 1; }
    fseek(f, 0, SEEK_END);
    long kernel_size = ftell(f);
    fseek(f, 0, SEEK_SET);
    uint8_t* kernel_data = (uint8_t*)malloc(kernel_size);
    if (fread(kernel_data, 1, kernel_size, f) != (size_t)kernel_size) {
        perror("Failed to read kernel"); return 1;
    }
    fclose(f);

    if (debug) printf("[GLAE] Kernel: %s (%ld bytes)\n", kernel_path, kernel_size);

    // Allocate guest DRAM on GPU
    uint8_t* d_dram;
    cudaMalloc(&d_dram, dram_size);
    cudaMemset(d_dram, 0, dram_size);

    // Copy kernel to DRAM at offset 0 (loaded at DRAM_BASE = 0x80000000)
    // Linux kernel Image is loaded at DRAM_BASE + 0x200000 for OpenSBI,
    // but without OpenSBI, we load at DRAM_BASE directly
    uint64_t kernel_offset = 0;
    if (kernel_size > 0x200000 || kernel_data[0] == 'M') {
        // Likely a full Linux Image — load at base
        kernel_offset = 0;
    }
    cudaMemcpy(d_dram + kernel_offset, kernel_data, kernel_size, cudaMemcpyHostToDevice);
    uint64_t entry_pc = DRAM_BASE + kernel_offset;

    if (debug) printf("[GLAE] Kernel loaded at 0x%llx\n", (unsigned long long)entry_pc);

    // Build and copy FDT
    uint8_t fdt_buf[8192];
    memset(fdt_buf, 0, sizeof(fdt_buf));
    const char* bootargs = "earlycon=uart8250,mmio,0x10000000,115200 console=ttyS0";
    int fdt_size = build_fdt(fdt_buf, DRAM_BASE, dram_size, bootargs);

    // Place DTB at end of DRAM minus 2MB (safe location)
    uint64_t dtb_offset = dram_size - 0x200000;
    cudaMemcpy(d_dram + dtb_offset, fdt_buf, fdt_size, cudaMemcpyHostToDevice);
    uint64_t dtb_addr = DRAM_BASE + dtb_offset;

    if (debug) printf("[GLAE] FDT: %d bytes at 0x%llx\n", fdt_size, (unsigned long long)dtb_addr);

    // Allocate ring buffers in pinned memory
    Ring* tx_ring;
    Ring* rx_ring;
    cudaMallocHost(&tx_ring, sizeof(Ring));
    cudaMallocHost(&rx_ring, sizeof(Ring));
    memset(tx_ring, 0, sizeof(Ring));
    memset(rx_ring, 0, sizeof(Ring));

    // Allocate device structs on GPU
    Uart* d_uart;
    cudaMalloc(&d_uart, sizeof(Uart));
    Uart h_uart = {};
    h_uart.tx_ring = tx_ring;
    h_uart.rx_ring = rx_ring;
    h_uart.lcr = 0x03;  // 8-N-1
    h_uart.iir = 0x01;  // no interrupt
    cudaMemcpy(d_uart, &h_uart, sizeof(Uart), cudaMemcpyHostToDevice);

    Clint* d_clint;
    cudaMalloc(&d_clint, sizeof(Clint));
    Clint h_clint = {};
    h_clint.mtimecmp = UINT64_MAX;
    cudaMemcpy(d_clint, &h_clint, sizeof(Clint), cudaMemcpyHostToDevice);

    Plic* d_plic;
    cudaMalloc(&d_plic, sizeof(Plic));
    cudaMemset(d_plic, 0, sizeof(Plic));

    // Machine struct
    Machine* d_mach;
    cudaMalloc(&d_mach, sizeof(Machine));
    Machine h_mach = { d_dram, dram_size, d_uart, d_clint, d_plic };
    cudaMemcpy(d_mach, &h_mach, sizeof(Machine), cudaMemcpyHostToDevice);

    // Hart state
    HartState* d_hart;
    cudaMalloc(&d_hart, sizeof(HartState));
    HartState h_hart;
    init_hart(&h_hart, entry_pc, dtb_addr, gpu_freq);
    cudaMemcpy(d_hart, &h_hart, sizeof(HartState), cudaMemcpyHostToDevice);

    // Set debug flag on device
    bool h_debug = debug;
    cudaMemcpyToSymbol(g_debug, &h_debug, sizeof(bool));

    // Set gpu_clock_base on device after sync
    cudaDeviceSynchronize();
    // We need to set gpu_clock_base on the device. Use a small kernel.
    // For simplicity, we set it to 0 and accept a time offset.
    // The kernel's clock starts from an arbitrary point.

    if (debug) printf("[GLAE] Starting execution...\n\n");

    // Non-blocking stdin for UART input
    set_nonblocking_stdin();

    // Save terminal settings
    struct termios old_term, new_term;
    tcgetattr(STDIN_FILENO, &old_term);
    new_term = old_term;
    new_term.c_lflag &= ~(ICANON | ECHO);
    tcsetattr(STDIN_FILENO, TCSANOW, &new_term);

    auto t_start = std::chrono::high_resolution_clock::now();
    uint64_t total_batches = 0;
    bool running = true;

    while (running) {
        vcpu_run<<<1, 1>>>(d_hart, d_mach);
        cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            fprintf(stderr, "[GLAE] CUDA error: %s\n", cudaGetErrorString(err));
            break;
        }
        total_batches++;

        // Drain UART TX
        uint8_t ch;
        while (ring_pop(tx_ring, &ch)) {
            putchar(ch);
        }
        fflush(stdout);

        // Feed UART RX from stdin
        char inbuf[16];
        int n = read(STDIN_FILENO, inbuf, sizeof(inbuf));
        if (n > 0) {
            for (int i = 0; i < n; i++)
                ring_push(rx_ring, (uint8_t)inbuf[i]);
        }

        // Check yield reason
        uint32_t yield;
        cudaMemcpy(&yield, &d_hart->yield_reason, sizeof(uint32_t), cudaMemcpyDeviceToHost);

        // Periodic progress report
        if (debug && (total_batches % 2000 == 0)) {
            uint64_t instret, pc;
            uint8_t priv;
            cudaMemcpy(&instret, &d_hart->instret, sizeof(uint64_t), cudaMemcpyDeviceToHost);
            cudaMemcpy(&pc, &d_hart->pc, sizeof(uint64_t), cudaMemcpyDeviceToHost);
            cudaMemcpy(&priv, &d_hart->priv, sizeof(uint8_t), cudaMemcpyDeviceToHost);
            auto now = std::chrono::high_resolution_clock::now();
            double elapsed = std::chrono::duration<double>(now - t_start).count();
            fprintf(stderr, "[GLAE] batch=%llu insns=%llu pc=%llx priv=%d yield=%u MIPS=%.1f (%.1fs)\n",
                    (unsigned long long)total_batches,
                    (unsigned long long)instret,
                    (unsigned long long)pc,
                    priv, yield,
                    instret / elapsed / 1e6, elapsed);
        }

        switch (yield) {
        case YIELD_UART_TX:
        case YIELD_BATCH_END:
            break; // continue
        case YIELD_WFI: {
            // Sleep briefly, then check for input or timer
            usleep(1000); // 1ms
            break;
        }
        case YIELD_HALT:
            running = false;
            break;
        case YIELD_FATAL:
            fprintf(stderr, "\n[GLAE] Fatal error\n");
            running = false;
            break;
        default:
            break;
        }
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(t_end - t_start).count();

    // Print stats
    if (debug) {
        uint64_t instret;
        cudaMemcpy(&instret, &d_hart->instret, sizeof(uint64_t), cudaMemcpyDeviceToHost);
        printf("\n[GLAE] === Execution Stats ===\n");
        printf("[GLAE] Instructions: %llu\n", (unsigned long long)instret);
        printf("[GLAE] Batches: %llu\n", (unsigned long long)total_batches);
        printf("[GLAE] Wall time: %.3f s\n", elapsed);
        printf("[GLAE] MIPS: %.2f\n", instret / elapsed / 1e6);
    }

    // Cleanup
    tcsetattr(STDIN_FILENO, TCSANOW, &old_term);
    cudaFreeHost(tx_ring);
    cudaFreeHost(rx_ring);
    cudaFree(d_hart);
    cudaFree(d_mach);
    cudaFree(d_uart);
    cudaFree(d_clint);
    cudaFree(d_plic);
    cudaFree(d_dram);
    free(kernel_data);

    return 0;
}
