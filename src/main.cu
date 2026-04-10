// ============================================================
// GLAE — GPU-native RV64GC Hypervisor
// Persistent kernel: one GPU block per hart, runs until halt.
// Host communicates via pinned memory only.
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
// Host-GPU control structure (pinned memory, visible to both)
// ============================================================
struct HostGpuCtrl {
    volatile uint32_t shutdown;              // host sets to 1 to stop all harts
    volatile uint32_t hart_halted[MAX_HARTS]; // hart sets to 1 when permanently halted
    volatile uint32_t fatal;                 // any hart sets to 1 on fatal error
    volatile uint64_t instret[MAX_HARTS];    // exported for host progress reporting
};

// ============================================================
// GPU Kernel — Persistent execution
// Each block = one hart. blockIdx.x = hart ID.
// Only thread 0 of each warp executes.
// Kernel runs until shutdown or all harts halt.
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

// Spin helper: wait ~N nanoseconds using clock64()
__device__ __forceinline__ void gpu_spin_ns(uint64_t ns, uint64_t gpu_freq) {
    uint64_t cycles = ns * gpu_freq / 1000000000ULL;
    uint64_t start = clock64();
    while (clock64() - start < cycles) {}
}

__global__ void __launch_bounds__(32, 1)
vcpu_run(HartState* harts, Machine* mach, HostGpuCtrl* ctrl) {
    const int hart_id = blockIdx.x;
    if (threadIdx.x != 0) return;

    HartState* hart = &harts[hart_id];

    // Initialize gpu_clock_base on first execution
    if (hart->gpu_clock_base == 0)
        hart->gpu_clock_base = clock64();

    // Per-hart instruction cache
    ICacheEntry* ic_base = hart_icache();

    // ---- Persistent event loop ----
    while (!ctrl->shutdown) {

        // HSM_STOPPED: spin-wait until started or shutdown
        if (hart->hsm_status == HSM_STOPPED) {
            // Use volatile to bypass L1 cache and see cross-SM writes
            volatile HartStatus* hsm = &hart->hsm_status;
            while (*hsm == HSM_STOPPED && !ctrl->shutdown)
                gpu_spin_ns(1000, hart->gpu_clock_freq);  // ~1us spin
            if (ctrl->shutdown) break;
        }

        // HSM_START_PENDING: initialize execution state
        if (hart->hsm_status == HSM_START_PENDING) {
            hart->pc = hart->hsm_start_addr;
            hart->x[10] = hart->mhartid;
            hart->x[11] = hart->hsm_start_arg;
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

        // ---- Execute a batch of instructions ----
        hart->yield_reason = YIELD_NONE;

        for (int i = 0; i < BATCH_SIZE && hart->yield_reason == YIELD_NONE; i++) {
            // Timer + PLIC check
            if ((i & 0x1FF) == 0) {
                clint_tick(hart, mach->clint);
                if (hart->stimecmp != UINT64_MAX && hart->get_mtime() >= hart->stimecmp)
                    hart->mip |= MIP_STIP;
                if (!(uart_compute_iir(mach->uart) & 0x01))
                    plic_set_pending(mach->plic, 10);
                plic_update_ext(hart, mach->plic, hart_id);
            }

            // Interrupt + WFI check
            if ((i & 0xFF) == 0) {
                if (hart->wfi) {
                    uint64_t pending = hart->mip & hart->mie;
                    if (pending) hart->wfi = 0;
                    else break;  // exit inner loop, handle WFI in outer loop
                }
                check_interrupts(hart);
            }

            // Fetch with icache
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

        // Export instret to pinned memory for host progress reporting
        ctrl->instret[hart_id] = hart->instret;

        // ---- Handle WFI inline: spin-wait until interrupt pending ----
        if (hart->wfi) {
            while (hart->wfi && !ctrl->shutdown) {
                // clock64() keeps ticking — mtime advances naturally
                clint_tick(hart, mach->clint);
                if (hart->stimecmp != UINT64_MAX && hart->get_mtime() >= hart->stimecmp)
                    hart->mip |= MIP_STIP;
                // Check UART RX → PLIC
                if (!(uart_compute_iir(mach->uart) & 0x01))
                    plic_set_pending(mach->plic, 10);
                plic_update_ext(hart, mach->plic, hart_id);

                uint64_t pending = hart->mip & hart->mie;
                if (pending) { hart->wfi = 0; break; }

                // Throttle spin to avoid hammering memory bus (~10us)
                gpu_spin_ns(10000, hart->gpu_clock_freq);
            }
            continue;  // resume execution
        }

        // ---- Handle halt/fatal ----
        if (hart->yield_reason == YIELD_HALT) {
            ctrl->hart_halted[hart_id] = 1;
            __threadfence_system();
            // Park: spin until shutdown
            while (!ctrl->shutdown)
                gpu_spin_ns(100000, hart->gpu_clock_freq);  // 100us
            break;
        }
        if (hart->yield_reason == YIELD_FATAL) {
            ctrl->fatal = 1;
            __threadfence_system();
            break;
        }

        // YIELD_UART_TX / YIELD_BATCH_END: just continue
        hart->yield_reason = YIELD_NONE;
    }
}

// ============================================================
// Host: SMP setup and polling loop
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
    h->mcounteren = 7;
    h->scounteren = 7;

    for (int i = 0; i < TLB_SIZE; i++) {
        h->itlb[i].valid = 0;
        h->dtlb[i].valid = 0;
    }

    if (is_boot_hart) {
        h->pc = entry_pc;
        h->priv = PRV_S;
        h->x[10] = hartid;
        h->x[11] = dtb_addr;
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

    // Build FDT
    uint8_t* fdt_buf = (uint8_t*)calloc(1, 65536);
    const char* bootargs = "earlycon=uart8250,mmio,0x10000000,115200 console=ttyS0 norandmaps";
    int fdt_size = build_fdt(fdt_buf, DRAM_BASE, dram_size, bootargs, num_harts);

    uint64_t dtb_offset = dram_size - 0x200000;
    if ((uint64_t)kernel_size > dtb_offset) {
        fprintf(stderr, "[GLAE] ERROR: kernel (%ld bytes) overlaps FDT\n", kernel_size);
        free(fdt_buf); free(kernel_data);
        return 1;
    }
    CUDA_CHECK(cudaMemcpy(d_dram + dtb_offset, fdt_buf, fdt_size, cudaMemcpyHostToDevice));
    uint64_t dtb_addr = DRAM_BASE + dtb_offset;
    free(fdt_buf);

    if (debug) printf("[GLAE] FDT: %d bytes at 0x%llx (%d CPUs)\n",
                      fdt_size, (unsigned long long)dtb_addr, num_harts);

    // Allocate ring buffers (pinned memory — shared between host and GPU)
    Ring* tx_ring;
    Ring* rx_ring;
    CUDA_CHECK(cudaMallocHost(&tx_ring, sizeof(Ring)));
    CUDA_CHECK(cudaMallocHost(&rx_ring, sizeof(Ring)));
    memset(tx_ring, 0, sizeof(Ring));
    memset(rx_ring, 0, sizeof(Ring));

    // Allocate host-GPU control (pinned memory)
    HostGpuCtrl* ctrl;
    CUDA_CHECK(cudaMallocHost(&ctrl, sizeof(HostGpuCtrl)));
    memset((void*)ctrl, 0, sizeof(HostGpuCtrl));

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

    // Allocate hart states
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

    if (debug) printf("[GLAE] Starting persistent kernel (%d harts)...\n\n", num_harts);

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
    auto t_last_report = t_start;

    // ---- Launch kernel ONCE — it runs until shutdown ----
    vcpu_run<<<num_harts, 32>>>(d_harts, d_mach, ctrl);

    // ---- Host polling loop (no cudaDeviceSynchronize) ----
    bool running = true;
    while (running) {
        // 1. Drain UART TX ring
        uint8_t ch;
        while (ring_pop(tx_ring, &ch)) putchar(ch);
        fflush(stdout);

        // 2. Feed UART RX ring
        char inbuf[16];
        int n = read(STDIN_FILENO, inbuf, sizeof(inbuf));
        if (n > 0) {
            for (int i = 0; i < n; i++)
                ring_push(rx_ring, (uint8_t)inbuf[i]);
        }

        // 3. Check for fatal error (pinned memory — no cudaMemcpy!)
        if (ctrl->fatal) {
            running = false;
            break;
        }

        // 4. Check if all harts permanently halted
        bool all_halted = true;
        for (int h = 0; h < num_harts; h++) {
            if (!ctrl->hart_halted[h]) { all_halted = false; break; }
        }
        if (all_halted) {
            running = false;
            break;
        }

        // 5. Progress report (every ~2 seconds)
        if (debug) {
            auto now = std::chrono::high_resolution_clock::now();
            double since_report = std::chrono::duration<double>(now - t_last_report).count();
            if (since_report >= 2.0) {
                t_last_report = now;
                uint64_t total_insns = 0;
                for (int h = 0; h < num_harts; h++)
                    total_insns += ctrl->instret[h];
                double elapsed = std::chrono::duration<double>(now - t_start).count();
                fprintf(stderr, "[GLAE] insns=%llu MIPS=%.1f (%.1fs) harts=%d\n",
                        (unsigned long long)total_insns,
                        total_insns / elapsed / 1e6, elapsed, num_harts);
            }
        }

        // 6. Brief yield — 100us polling interval
        usleep(100);
    }

    // Signal kernel to shut down and wait for all blocks to exit
    ctrl->shutdown = 1;
    cudaDeviceSynchronize();

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
    cudaFreeHost(ctrl);
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
