// ============================================================
// GLAE — GPU-native RV64GC Hypervisor
// SMP: one GPU block per hart, one warp per block
// ============================================================

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <unistd.h>
#include <fcntl.h>
#include <termios.h>
#include <signal.h>

// Platform
#include "platform/ring.h"
#include "platform/debug.cuh"

// Core
#include "core/icache.cuh"
#include "core/hart.cuh"

// ISA
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
// GPU Kernel — SMP execution
// Each block = one hart. blockIdx.x = hart ID.
// Only thread 0 of each warp executes.
// ============================================================

#define BATCH_SIZE 50000

#define CUDA_CHECK(call) do { \
    cudaError_t _err = (call); \
    if (_err != cudaSuccess) { \
        fprintf(stderr, "[GLAE] CUDA error: %s (%s:%d)\n", \
                cudaGetErrorString(_err), __FILE__, __LINE__); \
        exit(1); \
    } \
} while (0)

__global__ void __launch_bounds__(32, 1)
vcpu_run(HartState* harts, Machine* mach) {
    const int hart_id = blockIdx.x;
    if (threadIdx.x != 0) return;

    HartState* hart = &harts[hart_id];

    // Initialize gpu_clock_base on first execution
    if (hart->gpu_clock_base == 0)
        hart->gpu_clock_base = clock64();

    // Check HSM state — non-boot harts may be STOPPED
    if (hart->hsm_status == HSM_STOPPED) {
        hart->yield_reason = YIELD_WFI;
        return;
    }

    // Hart just started — initialize execution state
    if (hart->hsm_status == HSM_START_PENDING) {
        hart->pc = hart->hsm_start_addr;
        hart->x[10] = hart->mhartid;        // a0 = hartid
        hart->x[11] = hart->hsm_start_arg;  // a1 = opaque
        hart->priv = PRV_S;
        hart->hsm_status = HSM_STARTED;
        tlb_flush(hart->itlb);
        tlb_flush(hart->dtlb);
        icache_flush();
        DPRINTF("[HART%d] Started at pc=%llx a0=%llu a1=%llx\n",
                hart_id, (unsigned long long)hart->pc,
                (unsigned long long)hart->mhartid,
                (unsigned long long)hart->hsm_start_arg);
    }

    hart->yield_reason = YIELD_NONE;

    // Per-hart instruction cache
    ICacheEntry* ic_base = hart_icache();

    for (int i = 0; i < BATCH_SIZE && hart->yield_reason == YIELD_NONE; i++) {
        // Timer check
        if ((i & 0x1FF) == 0) {
            clint_tick(hart, mach->clint);
            if (hart->stimecmp != 0 && hart->get_mtime() >= hart->stimecmp)
                hart->mip |= MIP_STIP;
            // UART → PLIC: set IRQ 10 if UART has pending interrupt
            if (!(uart_compute_iir(mach->uart) & 0x01))
                plic_set_pending(mach->plic, 10);
            plic_update_ext(hart, mach->plic, hart_id);
        }

        // Interrupt check
        if ((i & 0xFF) == 0) {
            if (hart->wfi) {
                uint64_t pending = hart->mip & hart->mie;
                if (pending) hart->wfi = 0;
                else { hart->yield_reason = YIELD_WFI; break; }
            }
            check_interrupts(hart);
        }

        // Fetch with per-hart instruction cache
        uint32_t insn;
        int insn_len;
        uint64_t pc = hart->pc;
        uint32_t ic_idx = ((uint32_t)(pc >> 1)) & ICACHE_MASK;
        ICacheEntry* ic = &ic_base[ic_idx];

        if (ic->valid && ic->pc == pc) {
            insn = ic->insn;
            insn_len = ic->len;
        } else {
            uint32_t raw = 0;
            if (!mem_fetch(hart, mach, &raw)) continue;

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

            ic->pc = pc;
            ic->insn = insn;
            ic->len = (uint8_t)insn_len;
            ic->valid = 1;
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
// Host: SMP setup and main loop
// ============================================================

static void init_hart(HartState* h, int hartid, uint64_t entry_pc,
                      uint64_t dtb_addr, uint64_t gpu_freq, bool is_boot_hart) {
    memset(h, 0, sizeof(HartState));

    h->mhartid = hartid;
    h->misa = GLAE_MISA;

    // mstatus: FS=Initial(1), UXL=2, SXL=2
    h->mstatus = (1ULL << MSTATUS_FS_SHIFT) | (2ULL << 32) | (2ULL << 34);

    // Delegate standard traps and interrupts to S-mode
    h->medeleg = (1 << 0) | (1 << 1) | (1 << 2) | (1 << 3) |
                 (1 << 4) | (1 << 5) | (1 << 6) | (1 << 7) |
                 (1 << 8) | (1 << 12) | (1 << 13) | (1 << 15);
    h->mideleg = MIP_SSIP | MIP_STIP | MIP_SEIP;

    h->gpu_clock_freq = gpu_freq;
    h->stimecmp = UINT64_MAX;
    h->mcounteren = 7;  // CY, TM, IR: allow S-mode counter access
    h->scounteren = 7;  // Allow U-mode counter access

    for (int i = 0; i < TLB_SIZE; i++) {
        h->itlb[i].valid = 0;
        h->dtlb[i].valid = 0;
    }

    if (is_boot_hart) {
        h->pc = entry_pc;
        h->priv = PRV_S;
        h->x[10] = hartid;   // a0 = hartid
        h->x[11] = dtb_addr; // a1 = dtb address
        h->hsm_status = HSM_STARTED;
    } else {
        h->pc = 0;
        h->priv = PRV_S;
        h->hsm_status = HSM_STOPPED;
    }
}

static void set_nonblocking_stdin() {
    int flags = fcntl(STDIN_FILENO, F_GETFL, 0);
    fcntl(STDIN_FILENO, F_SETFL, flags | O_NONBLOCK);
}

// Terminal restore on abnormal exit
static struct termios g_old_term;
static bool g_term_modified = false;

static void restore_terminal() {
    if (g_term_modified)
        tcsetattr(STDIN_FILENO, TCSANOW, &g_old_term);
}

static void signal_handler(int sig) {
    restore_terminal();
    _exit(128 + sig);
}

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <kernel-image> [--smp N] [--dram-mb N]\n", argv[0]);
        return 1;
    }

    const char* kernel_path = argv[1];
    uint64_t dram_size = DRAM_SIZE_DEFAULT;
    int num_harts = 1;

    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "--dram-mb") == 0 && i + 1 < argc)
            dram_size = (uint64_t)atoi(argv[++i]) * 1024 * 1024;
        else if (strcmp(argv[i], "--smp") == 0 && i + 1 < argc)
            num_harts = atoi(argv[++i]);
    }

    if (num_harts < 1) num_harts = 1;
    if (num_harts > MAX_HARTS) num_harts = MAX_HARTS;

    bool debug = getenv("GLAE_DEBUG") != nullptr;

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    uint64_t gpu_freq = (uint64_t)prop.clockRate * 1000;

    if (debug) {
        printf("[GLAE] GPU: %s (%d SMs, %llu Hz clock)\n",
               prop.name, prop.multiProcessorCount, (unsigned long long)gpu_freq);
        printf("[GLAE] SMP: %d harts\n", num_harts);
        printf("[GLAE] DRAM: %llu MB at 0x%llx\n",
               (unsigned long long)(dram_size / (1024*1024)),
               (unsigned long long)DRAM_BASE);
    }

    // Load kernel
    FILE* f = fopen(kernel_path, "rb");
    if (!f) { perror("Failed to open kernel"); return 1; }
    fseek(f, 0, SEEK_END);
    long kernel_size = ftell(f);
    fseek(f, 0, SEEK_SET);
    uint8_t* kernel_data = (uint8_t*)malloc(kernel_size);
    if (fread(kernel_data, 1, kernel_size, f) != (size_t)kernel_size) {
        fprintf(stderr, "Failed to read kernel\n");
        free(kernel_data); fclose(f); return 1;
    }
    fclose(f);

    if (debug) printf("[GLAE] Kernel: %s (%ld bytes)\n", kernel_path, kernel_size);

    // Allocate guest DRAM
    uint8_t* d_dram;
    CUDA_CHECK(cudaMalloc(&d_dram, dram_size));
    CUDA_CHECK(cudaMemset(d_dram, 0, dram_size));

    uint64_t kernel_offset = 0;
    CUDA_CHECK(cudaMemcpy(d_dram + kernel_offset, kernel_data, kernel_size, cudaMemcpyHostToDevice));
    uint64_t entry_pc = DRAM_BASE + kernel_offset;

    // Build FDT with N CPUs
    uint8_t* fdt_buf = (uint8_t*)calloc(1, 65536);  // large enough for many CPUs
    const char* bootargs = "earlycon=uart8250,mmio,0x10000000,115200 console=ttyS0 norandmaps";
    int fdt_size = build_fdt(fdt_buf, DRAM_BASE, dram_size, bootargs, num_harts);

    uint64_t dtb_offset = dram_size - 0x200000;
    if ((uint64_t)kernel_size > dtb_offset) {
        fprintf(stderr, "[GLAE] ERROR: kernel (%ld bytes) overlaps FDT at offset 0x%llx\n",
                kernel_size, (unsigned long long)dtb_offset);
        free(fdt_buf); free(kernel_data);
        return 1;
    }
    CUDA_CHECK(cudaMemcpy(d_dram + dtb_offset, fdt_buf, fdt_size, cudaMemcpyHostToDevice));
    uint64_t dtb_addr = DRAM_BASE + dtb_offset;
    free(fdt_buf);

    if (debug) printf("[GLAE] FDT: %d bytes at 0x%llx (%d CPUs)\n",
                      fdt_size, (unsigned long long)dtb_addr, num_harts);

    // Allocate ring buffers
    Ring* tx_ring;
    Ring* rx_ring;
    CUDA_CHECK(cudaMallocHost(&tx_ring, sizeof(Ring)));
    CUDA_CHECK(cudaMallocHost(&rx_ring, sizeof(Ring)));
    memset(tx_ring, 0, sizeof(Ring));
    memset(rx_ring, 0, sizeof(Ring));

    // Allocate devices
    Uart* d_uart;
    CUDA_CHECK(cudaMalloc(&d_uart, sizeof(Uart)));
    Uart h_uart = {};
    h_uart.tx_ring = tx_ring;
    h_uart.rx_ring = rx_ring;
    h_uart.lcr = 0x03;
    h_uart.iir = 0x01;
    CUDA_CHECK(cudaMemcpy(d_uart, &h_uart, sizeof(Uart), cudaMemcpyHostToDevice));

    Clint* d_clint;
    CUDA_CHECK(cudaMalloc(&d_clint, sizeof(Clint)));
    {
        Clint h_clint = {};
        for (int i = 0; i < MAX_HARTS; i++) h_clint.mtimecmp[i] = UINT64_MAX;
        CUDA_CHECK(cudaMemcpy(d_clint, &h_clint, sizeof(Clint), cudaMemcpyHostToDevice));
    }

    Plic* d_plic;
    CUDA_CHECK(cudaMalloc(&d_plic, sizeof(Plic)));
    CUDA_CHECK(cudaMemset(d_plic, 0, sizeof(Plic)));

    // Allocate N hart states
    HartState* d_harts;
    CUDA_CHECK(cudaMalloc(&d_harts, num_harts * sizeof(HartState)));
    {
        HartState* h_harts = (HartState*)calloc(num_harts, sizeof(HartState));
        for (int i = 0; i < num_harts; i++)
            init_hart(&h_harts[i], i, entry_pc, dtb_addr, gpu_freq, i == 0);
        CUDA_CHECK(cudaMemcpy(d_harts, h_harts, num_harts * sizeof(HartState), cudaMemcpyHostToDevice));
        free(h_harts);
    }

    // Allocate per-hart instruction caches
    ICacheEntry* d_icache_pool;
    CUDA_CHECK(cudaMalloc(&d_icache_pool, num_harts * ICACHE_ENTRIES * sizeof(ICacheEntry)));
    CUDA_CHECK(cudaMemset(d_icache_pool, 0, num_harts * ICACHE_ENTRIES * sizeof(ICacheEntry)));
    CUDA_CHECK(cudaMemcpyToSymbol(g_icache_pool, &d_icache_pool, sizeof(ICacheEntry*)));
    CUDA_CHECK(cudaMemcpyToSymbol(g_num_harts, &num_harts, sizeof(int)));

    // Machine struct
    Machine* d_mach;
    CUDA_CHECK(cudaMalloc(&d_mach, sizeof(Machine)));
    Machine h_mach = { d_dram, dram_size, d_uart, d_clint, d_plic, d_harts, num_harts };
    CUDA_CHECK(cudaMemcpy(d_mach, &h_mach, sizeof(Machine), cudaMemcpyHostToDevice));

    // Debug flag
    bool h_debug = debug;
    CUDA_CHECK(cudaMemcpyToSymbol(g_debug, &h_debug, sizeof(bool)));

    CUDA_CHECK(cudaDeviceSynchronize());

    if (debug) printf("[GLAE] Starting execution (%d harts)...\n\n", num_harts);

    set_nonblocking_stdin();
    tcgetattr(STDIN_FILENO, &g_old_term);
    struct termios new_term = g_old_term;
    new_term.c_lflag &= ~(ICANON | ECHO);
    tcsetattr(STDIN_FILENO, TCSANOW, &new_term);
    g_term_modified = true;
    atexit(restore_terminal);
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);

    auto t_start = std::chrono::high_resolution_clock::now();
    uint64_t total_batches = 0;
    bool running = true;

    while (running) {
        // Launch one block per hart, 32 threads per block
        vcpu_run<<<num_harts, 32>>>(d_harts, d_mach);
        cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            fprintf(stderr, "[GLAE] CUDA error: %s\n", cudaGetErrorString(err));
            break;
        }
        total_batches++;

        // Drain UART TX
        uint8_t ch;
        while (ring_pop(tx_ring, &ch)) putchar(ch);
        fflush(stdout);

        // Feed UART RX
        char inbuf[16];
        int n = read(STDIN_FILENO, inbuf, sizeof(inbuf));
        if (n > 0) {
            for (int i = 0; i < n; i++)
                ring_push(rx_ring, (uint8_t)inbuf[i]);
        }

        // Check if ALL harts are halted
        bool all_halted = true;
        for (int h = 0; h < num_harts; h++) {
            uint32_t yield;
            cudaMemcpy(&yield, &d_harts[h].yield_reason, sizeof(uint32_t), cudaMemcpyDeviceToHost);
            if (yield == YIELD_FATAL) { running = false; break; }
            if (yield != YIELD_HALT && yield != YIELD_WFI) all_halted = false;
        }

        // Check if any STOPPED hart has been START_PENDING'd
        // (no action needed — the kernel handles it on next launch)

        // If all harts are WFI/halted, brief sleep
        if (all_halted) {
            // Check if any hart can be woken
            bool any_wfi = false;
            for (int h = 0; h < num_harts; h++) {
                uint32_t yield;
                cudaMemcpy(&yield, &d_harts[h].yield_reason, sizeof(uint32_t), cudaMemcpyDeviceToHost);
                if (yield == YIELD_WFI) any_wfi = true;
                if (yield == YIELD_HALT) {
                    HartStatus status;
                    cudaMemcpy(&status, &d_harts[h].hsm_status, sizeof(HartStatus), cudaMemcpyDeviceToHost);
                    if (status == HSM_START_PENDING) any_wfi = true; // will start next batch
                }
            }
            if (any_wfi) {
                usleep(1000);
                // Advance guest time for WFI harts: clock64() doesn't tick while
                // the GPU kernel isn't running, so subtract from gpu_clock_base
                // to simulate 1ms of elapsed time per sleep cycle.
                uint64_t advance = gpu_freq / 1000;  // 1ms worth of GPU cycles
                for (int h = 0; h < num_harts; h++) {
                    uint32_t yield;
                    cudaMemcpy(&yield, &d_harts[h].yield_reason, sizeof(uint32_t), cudaMemcpyDeviceToHost);
                    if (yield == YIELD_WFI) {
                        uint64_t base;
                        cudaMemcpy(&base, &d_harts[h].gpu_clock_base, sizeof(uint64_t), cudaMemcpyDeviceToHost);
                        base -= advance;
                        cudaMemcpy(&d_harts[h].gpu_clock_base, &base, sizeof(uint64_t), cudaMemcpyHostToDevice);
                    }
                }
            }
            else { running = false; }  // all halted permanently
        }

        // Progress report
        if (debug && (total_batches % 2000 == 0)) {
            uint64_t total_insns = 0;
            for (int h = 0; h < num_harts; h++) {
                uint64_t instret;
                cudaMemcpy(&instret, &d_harts[h].instret, sizeof(uint64_t), cudaMemcpyDeviceToHost);
                total_insns += instret;
            }
            auto now = std::chrono::high_resolution_clock::now();
            double elapsed = std::chrono::duration<double>(now - t_start).count();
            fprintf(stderr, "[GLAE] batch=%llu total_insns=%llu MIPS=%.1f (%.1fs) harts=%d\n",
                    (unsigned long long)total_batches,
                    (unsigned long long)total_insns,
                    total_insns / elapsed / 1e6, elapsed, num_harts);
        }
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(t_end - t_start).count();

    if (debug) {
        uint64_t total_insns = 0;
        for (int h = 0; h < num_harts; h++) {
            uint64_t instret;
            cudaMemcpy(&instret, &d_harts[h].instret, sizeof(uint64_t), cudaMemcpyDeviceToHost);
            total_insns += instret;
            if (instret > 0)
                printf("[GLAE] Hart %d: %llu instructions\n", h, (unsigned long long)instret);
        }
        printf("[GLAE] Total: %llu instructions, %.3f s, %.2f MIPS\n",
               (unsigned long long)total_insns, elapsed, total_insns / elapsed / 1e6);
    }

    restore_terminal();
    cudaFreeHost(tx_ring);
    cudaFreeHost(rx_ring);
    cudaFree(d_harts);
    cudaFree(d_mach);
    cudaFree(d_uart);
    cudaFree(d_clint);
    cudaFree(d_plic);
    cudaFree(d_dram);
    cudaFree(d_icache_pool);
    free(kernel_data);

    return 0;
}
