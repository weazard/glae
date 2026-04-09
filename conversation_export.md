# GPU-Accelerated Computation & Emulation — Full Conversation Log

**Date:** April 9, 2026  
**Participants:** Michael, Claude (Opus 4.6)  
**Platform:** claude.ai  
**Attachments:** `riscv-unprivileged.pdf`, `riscv-privileged.pdf` (uploaded at end, not included in export)

---

## 1. Initial Code Review

**Michael** uploaded a CUDA file (`gpu_emu.cu`) implementing a GPU-accelerated emulation proof of concept — a simple RISC-like VM where instruction decoding is offloaded to the GPU while execution remains on the CPU. He also provided the runtime output:

### Michael's Question

> does this prove the attached approach viable?

---

## 2. Claude's Analysis

**Claude identified three fundamental problems:**

1. **Results are wrong.** Fibonacci(5) produced 103 instead of 5. The encoding function OR'd `rs2` and `imm` into overlapping bit positions, so `ADD r1, r2, r0` used register 0 (which held `n=5`) instead of acting as a move.

2. **Speedups are noise.** Microsecond-scale timings on tiny programs are dominated by measurement jitter, cache effects, and CUDA runtime overhead.

3. **The approach doesn't test what matters.** Pre-decoding instructions on GPU then executing sequentially on CPU offloads the cheapest part of emulation (bit shifts) while ignoring the expensive part (execution). The PCIe transfer overhead alone exceeds the decode savings.

**Claude's recommendation:** To prove GPU-accelerated emulation viable, the code needed to tackle parallel execution of independent instruction streams — many VM instances running simultaneously, one per GPU thread.

---

## 3. Corrected Implementation

**Michael:** "your task is to correct it then"

### Claude's Reasoning

The fix required three changes.

---

## 4. Discussion: Why Does GPU Help Here?

**Michael:** "but why does it show speedups when gpu should only be really useful with video/ai tasks not computation"

**Claude:** GPUs aren't limited to video/AI. A GPU is thousands of simple cores running the same code in parallel. Video and AI happen to be embarrassingly parallel (millions of independent operations), but the same pattern applies here: 50,000 independent VMs, each assigned to one GPU thread, no communication between them.

The GPU wins at scale because a CPU processes 50,000 VMs sequentially (or across 8-16 cores) while the GPU runs them all simultaneously. Individual VM execution is *slower* per GPU core (lower clock, simpler pipeline), which is why at 100 VMs the CPU wins. The crossover happens when parallelism compensates for per-core weakness.

---

## 5. Can You Parallelize a Single VM?

**Michael:** "but can't you split each thread into thousands of threads and delegate to gpu"

**Claude:** Not really. Each instruction depends on the previous one:

```
r1 = r2 + r3      ← need r2, r3 from prior state
r4 = r1 * r5      ← need r1 from the line above
beq r4, r0, +5    ← need r4 to know WHERE to go next
```

You can't execute instruction 1000 until you've executed 0–999, because you don't know register values or even which instruction 1000 *is* (branches change the path). This is a serial dependency chain — the thing GPUs are worst at.

Research approaches include speculative execution (guess branch direction, discard wrong path), basic block parallelism (find independent blocks), and SIMD across data (recognize array operations). But CPUs already do speculative execution in hardware better than GPU software could, dependency chains in real programs are long, and SIMD isn't general emulation.

**Bottom line:** A single thread of execution is inherently serial. GPUs help emulation only by running many independent VMs in parallel.

---

## 6. Making GPU Beat CPU at Computation Generally

**Michael:** "any strategies to amend this and make it practical"

**Claude identified five approaches:**

1. **JIT recompilation on GPU** — translate guest code blocks into native GPU operations ahead of time
2. **Memory subsystem offload** — CPU executes, GPU handles virtual memory (TLB, page tables, MMIO)
3. **Peripheral/device emulation** — each device as an independent GPU-thread state machine
4. **Ahead-of-time dependency analysis** — find instruction-level parallelism across basic blocks
5. **Hybrid approach** (practical sweet spot) — CPU handles serial control flow, GPU accelerates bulk operations the guest triggers

---

## 7. Can GPU Actually Beat CPU at Serial Computation?

**Michael:** "i'm talking about making gpu more effective at computing than cpu, not necessarily in context of vms"

**Claude:** For serial computation, probably not — it's a hardware architecture problem. CPU cores are optimized for single-thread speed (deep pipelines, branch prediction, out-of-order execution, speculative execution, large per-core caches). GPU cores have almost none of that.

**The real question:** Can you restructure computation itself to be more parallel?

- **Dataflow architectures** — execute any instruction whose inputs are ready (like ML framework DAGs)
- **Functional programming models** — pure functions are inherently parallelizable
- **Cellular automata / spatial computing** — every memory cell is a tiny processor

**The uncomfortable truth:** Most useful computation is control-flow-heavy and serial because that's how humans think. The gap isn't GPU vs CPU — it's serial thinking vs parallel thinking. The hardware is there; the algorithms are the bottleneck.

---

## 8. The Challenge: Prove It

**Michael:** "so the question is whether it's worthwhile to rely on that kinda structure being faster than the more traditional ideas. To prove that we'll have to take a computation task that CPU handles better and design an approach that utilizes gpu instead. Preferably something that wouldn't be close anyway... something complex that involves what cpus are proficient at like deep pipelines, branch prediction, out-of-order execution, speculative execution, large caches per core"

### Claude's Reasoning

The ideal candidate is **DFA (Deterministic Finite Automaton) simulation** — processing a string one character at a time where each state depends on the previous. It's the textbook serial problem: tight loop, no visible parallelism, branch predictor paradise, working set fits in L1 cache.

The algebraic trick: each character's effect can be represented as a *transition function* (a small array mapping old_state → new_state). Two transition functions compose into a third, and composition is **associative**. This means you can parallel-reduce them — split the string across thousands of GPU threads, each composes locally, then tree-reduce.

The GPU does ~7x more total work per character (composing 7-element arrays vs single table lookups), but spreads it across thousands of cores. At sufficient input sizes, parallelism should overwhelm the extra work.

### Parallel DFA Implementation (`parallel_dfa.cu`)

```cuda
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <string.h>

// ============================================================
// The Problem: DFA simulation (textbook serial computation)
//
// Given a binary string, determine if the number is divisible by 7.
// DFA has 7 states (remainders 0-6).
//   On '0': state = (2 * state) % 7
//   On '1': state = (2 * state + 1) % 7
//
// CPU approach: tight loop, one lookup per char. This is what CPUs
// are literally designed for — sequential dependent state updates,
// branch prediction, L1-resident lookup table, deep pipeline.
//
// GPU approach: decompose each character into a TRANSITION FUNCTION
// (a map from state -> state). Transition functions compose
// associatively, so we can parallel-reduce them.
// ============================================================

#define NS 7  // DFA states (remainder mod 7)

struct Trans {
    uint8_t t[NS];
};

__host__ __device__ Trans make_identity() {
    Trans id;
    for (int i = 0; i < NS; i++) id.t[i] = i;
    return id;
}

__host__ __device__ Trans compose(Trans a, Trans b) {
    Trans c;
    for (int i = 0; i < NS; i++) c.t[i] = b.t[a.t[i]];
    return c;
}

// ============================================================
// CPU: Optimized serial DFA
// ============================================================
int cpu_dfa(const unsigned char* input, size_t n) {
    uint8_t table[NS][256];
    for (int s = 0; s < NS; s++) {
        for (int c = 0; c < 256; c++) {
            if (c == '0')      table[s][c] = (2 * s) % NS;
            else if (c == '1') table[s][c] = (2 * s + 1) % NS;
            else               table[s][c] = s;
        }
    }

    int state = 0;
    for (size_t i = 0; i < n; i++) {
        state = table[state][input[i]];
    }
    return state;
}

int cpu_dfa_branchy(const unsigned char* input, size_t n) {
    int state = 0;
    for (size_t i = 0; i < n; i++) {
        if (input[i] == '0')
            state = (2 * state) % NS;
        else if (input[i] == '1')
            state = (2 * state + 1) % NS;
    }
    return state;
}

// ============================================================
// GPU: Parallel transition-function composition
// ============================================================

__constant__ Trans d_char_trans[256];

void init_char_trans() {
    Trans h_table[256];
    for (int c = 0; c < 256; c++) {
        if (c == '0') {
            for (int s = 0; s < NS; s++) h_table[c].t[s] = (2 * s) % NS;
        } else if (c == '1') {
            for (int s = 0; s < NS; s++) h_table[c].t[s] = (2 * s + 1) % NS;
        } else {
            h_table[c] = make_identity();
        }
    }
    cudaMemcpyToSymbol(d_char_trans, h_table, sizeof(h_table));
}

#define BLOCK_SIZE 256

__global__ void map_reduce_kernel(const unsigned char* input, Trans* block_results,
                                  size_t n, int chunk) {
    int tid = threadIdx.x;
    size_t global_id = (size_t)blockIdx.x * BLOCK_SIZE + tid;
    size_t start = global_id * chunk;

    Trans local = make_identity();
    size_t end = start + chunk;
    if (end > n) end = n;
    for (size_t i = start; i < end; i++) {
        local = compose(local, d_char_trans[input[i]]);
    }

    // Order-preserving tree reduction (composition is non-commutative!)
    __shared__ Trans smem[BLOCK_SIZE];
    smem[tid] = local;
    __syncthreads();

    for (int stride = 1; stride < BLOCK_SIZE; stride <<= 1) {
        int idx = 2 * stride * tid;
        if (idx + stride < BLOCK_SIZE) {
            smem[idx] = compose(smem[idx], smem[idx + stride]);
        }
        __syncthreads();
    }

    if (tid == 0) block_results[blockIdx.x] = smem[0];
}

int gpu_dfa(const unsigned char* d_input, size_t n, double* kernel_ms) {
    int chunk = 512;
    int total_threads = (n + chunk - 1) / chunk;
    int n_blocks = (total_threads + BLOCK_SIZE - 1) / BLOCK_SIZE;

    Trans* d_block_results;
    cudaMalloc(&d_block_results, n_blocks * sizeof(Trans));

    // Warmup
    map_reduce_kernel<<<n_blocks, BLOCK_SIZE>>>(d_input, d_block_results, n, chunk);
    cudaDeviceSynchronize();

    // Timed run
    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);
    cudaEventRecord(t0);

    map_reduce_kernel<<<n_blocks, BLOCK_SIZE>>>(d_input, d_block_results, n, chunk);

    cudaEventRecord(t1);
    cudaEventSynchronize(t1);
    float ms;
    cudaEventElapsedTime(&ms, t0, t1);
    *kernel_ms = ms;

    cudaEventDestroy(t0);
    cudaEventDestroy(t1);

    Trans* h_results = new Trans[n_blocks];
    cudaMemcpy(h_results, d_block_results, n_blocks * sizeof(Trans), cudaMemcpyDeviceToHost);
    cudaFree(d_block_results);

    Trans total = make_identity();
    for (int i = 0; i < n_blocks; i++) {
        total = compose(total, h_results[i]);
    }
    delete[] h_results;

    return total.t[0];
}

// ============================================================
// Benchmark
// ============================================================
int main() {
    printf("=============================================================\n");
    printf("  Parallel DFA: GPU vs CPU on Inherently Serial Computation\n");
    printf("=============================================================\n\n");

    printf("Task: determine if a binary string represents a number\n");
    printf("divisible by 7. This is a 7-state DFA — the canonical\n");
    printf("example of serial state-dependent computation.\n\n");

    printf("CPU advantage: tight loop, no branches (lookup table),\n");
    printf("L1-resident working set, deep pipeline, 1 cycle/char.\n\n");

    printf("GPU trick: decompose into transition functions and\n");
    printf("parallel-reduce using associative composition.\n\n");

    init_char_trans();

    // Warmup GPU
    {
        unsigned char* d_tmp;
        cudaMalloc(&d_tmp, 1024);
        cudaFree(d_tmp);
    }

    size_t sizes[] = {1000000, 10000000, 100000000, 500000000};
    const char* labels[] = {"1M", "10M", "100M", "500M"};

    for (int si = 0; si < 4; si++) {
        size_t n = sizes[si];
        printf("--- Input: %s characters ---\n", labels[si]);

        unsigned char* h_input = (unsigned char*)malloc(n);
        if (!h_input) { printf("  malloc failed\n\n"); continue; }
        uint32_t rng = 42;
        for (size_t i = 0; i < n; i++) {
            rng = rng * 1664525u + 1013904223u;
            h_input[i] = '0' + ((rng >> 16) & 1);
        }

        // CPU branchy
        auto t0 = std::chrono::high_resolution_clock::now();
        int cpu_branchy_result = cpu_dfa_branchy(h_input, n);
        auto t1 = std::chrono::high_resolution_clock::now();
        double cpu_branchy_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        // CPU optimized
        t0 = std::chrono::high_resolution_clock::now();
        int cpu_result = cpu_dfa(h_input, n);
        t1 = std::chrono::high_resolution_clock::now();
        double cpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        if (cpu_result != cpu_branchy_result) {
            printf("  CPU MISMATCH: branchy=%d lookup=%d\n", cpu_branchy_result, cpu_result);
        }

        // GPU
        unsigned char* d_input;
        cudaMalloc(&d_input, n);
        cudaMemcpy(d_input, h_input, n, cudaMemcpyHostToDevice);
        cudaDeviceSynchronize();

        double gpu_kernel_ms;
        int gpu_result = gpu_dfa(d_input, n, &gpu_kernel_ms);

        t0 = std::chrono::high_resolution_clock::now();
        cudaMemcpy(d_input, h_input, n, cudaMemcpyHostToDevice);
        double gpu_with_transfer_ms;
        int gpu_result2 = gpu_dfa(d_input, n, &gpu_with_transfer_ms);
        t1 = std::chrono::high_resolution_clock::now();
        double gpu_total_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        bool match = (cpu_result == gpu_result);

        printf("  CPU (branchy):     %8.2f ms\n", cpu_branchy_ms);
        printf("  CPU (lookup tbl):  %8.2f ms   <-- best serial\n", cpu_ms);
        printf("  GPU (kernel only): %8.2f ms   speedup vs best CPU: %5.1fx  %s\n",
               gpu_kernel_ms, cpu_ms / gpu_kernel_ms, match ? "MATCH" : "MISMATCH");
        printf("  GPU (incl H2D):    %8.2f ms   speedup vs best CPU: %5.1fx\n",
               gpu_total_ms, cpu_ms / gpu_total_ms);
        printf("\n");

        cudaFree(d_input);
        free(h_input);
    }

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("GPU: %s (%d SMs, %.0f MHz, %.1f GB)\n",
           prop.name, prop.multiProcessorCount,
           prop.clockRate / 1000.0, prop.totalGlobalMem / 1e9);
    printf("Bandwidth: %.0f GB/s\n\n", 2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1e6);

    printf("=============================================================\n");
    printf("Key insight: The serial DFA loop is EXACTLY what CPUs excel at.\n");
    printf("No branches, L1-resident table, 1 cycle/char, deep pipeline.\n");
    printf("Yet the GPU wins by exploiting an algebraic property:\n");
    printf("transition functions compose associatively, enabling parallel\n");
    printf("reduction over an operation that APPEARS inherently serial.\n");
    printf("\n");
    printf("This is not 'many independent tasks.' This is ONE serial\n");
    printf("computation, restructured to unlock parallelism.\n");
    printf("=============================================================\n");

    return 0;
}
```

### Initial Output (with a bug)

```
--- Input: 1M characters ---
  GPU (kernel only):     0.13 ms   speedup vs best CPU:  15.3x  MISMATCH
--- Input: 10M characters ---
  GPU (kernel only):     0.13 ms   speedup vs best CPU: 153.5x  MISMATCH
--- Input: 100M characters ---
  GPU (kernel only):     0.85 ms   speedup vs best CPU: 227.9x  MISMATCH
--- Input: 500M characters ---
  GPU (kernel only):     4.00 ms   speedup vs best CPU: 243.8x  MATCH
```

### Bug: Non-Commutative Reduction

**Claude's analysis:** The standard halving tree reduction (`smem[tid] = compose(smem[tid], smem[tid+stride])`) pairs thread 0 with thread 128, producing order `T0, T128, T64, T192...` — completely scrambled. Function composition is **not commutative**, so this gives wrong results. The 500M match was coincidental.

**Fix:** Order-preserving reduction that pairs adjacent elements:

```cuda
// WRONG (scrambles order for non-commutative ops):
for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
        smem[tid] = compose(smem[tid], smem[tid + stride]);
    }
    __syncthreads();
}

// CORRECT (preserves left-to-right composition order):
for (int stride = 1; stride < BLOCK_SIZE; stride <<= 1) {
    int idx = 2 * stride * tid;
    if (idx + stride < BLOCK_SIZE) {
        smem[idx] = compose(smem[idx], smem[idx + stride]);
    }
    __syncthreads();
}
```

### Final Corrected Output

```
=============================================================
  Parallel DFA: GPU vs CPU on Inherently Serial Computation
=============================================================

--- Input: 1M characters ---
  CPU (branchy):         3.95 ms
  CPU (lookup tbl):      1.96 ms   <-- best serial
  GPU (kernel only):     0.13 ms   speedup vs best CPU:  15.3x  MATCH
  GPU (incl H2D):        0.33 ms   speedup vs best CPU:   5.9x

--- Input: 10M characters ---
  CPU (branchy):        39.01 ms
  CPU (lookup tbl):     19.61 ms   <-- best serial
  GPU (kernel only):     0.13 ms   speedup vs best CPU: 154.2x  MATCH
  GPU (incl H2D):        0.86 ms   speedup vs best CPU:  22.9x

--- Input: 100M characters ---
  CPU (branchy):       385.76 ms
  CPU (lookup tbl):    195.50 ms   <-- best serial
  GPU (kernel only):     0.85 ms   speedup vs best CPU: 230.1x  MATCH
  GPU (incl H2D):        7.20 ms   speedup vs best CPU:  27.2x

--- Input: 500M characters ---
  CPU (branchy):      1913.33 ms
  CPU (lookup tbl):    979.73 ms   <-- best serial
  GPU (kernel only):     3.95 ms   speedup vs best CPU: 248.3x  MATCH
  GPU (incl H2D):       52.39 ms   speedup vs best CPU:  18.7x

GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition (188 SMs, 2430 MHz, 102.0 GB)
Bandwidth: 1598 GB/s
```

---

## 9. CPU Threading Fairness

**Michael:** "does the cpu part only utilize one core or multithreaded"

**Claude:** Single core — a plain `for` loop, no threads, no SIMD. An OpenMP version with the same transition-composition trick across CPU cores could get 16-32x over single-core, bringing CPU to ~30-60ms for 500M. GPU at 4ms would still win ~10x. The single-core comparison arguably proves the thesis better though — the point is that algebraic restructuring unlocks parallelism in apparently serial computation, and GPU just has more parallel resources to exploit it.

---

## 10. Prior Art

**Michael:** "what was the inspiration for the gpu implementation? Is it based on existing research/practices or completely original?"

**Claude:** It's well-established theory:

- **Parallel prefix / scan** — Blelloch (1990). Associative operations can be parallel-reduced.
- **Monoid homomorphisms** — Bird and Meertens (1980s-90s), the "BMF" / Squiggol school of functional programming.
- **Parallel DFA specifically** — Ladner & Fischer (1980) on parallel prefix for automata; Mytkowicz, Musuvathi & Schulte (Microsoft Research, 2014) "Data-Parallel Finite-State Machines" — essentially this exact implementation.
- **GPU text processing** — NVIDIA's nvParse, the cuDF library in RAPIDS for parallel CSV field detection.

The implementation assembled existing pieces. Nothing original — the value was the concrete demonstration.

---

## 11. Toward Something New: GPU-Native Hypervisor

**Michael:** "Right, that has its application but somewhat defeats the principle of this exercise. If it's already established theory that means that we aren't finding anything new. Let's try implementing a somewhat complex hypervisor... Design it with the gpu computation in mind, relying on cpu as little as possible."

**Claude proposed a GPU-native RISC-V hypervisor:**

**Entirely on GPU:**
- vCPU execution (one GPU thread = one vCPU)
- Instruction fetch / decode / execute (RV32I)
- MMU with page tables (guest-controlled)
- CSR registers (machine/supervisor control)
- Trap handling (ecalls, page faults, illegal instructions)
- Timer interrupts via cycle counting
- Inter-vCPU shared memory (guest SMP)

**CPU only touches:**
- Loading guest ELF images into GPU memory before launch
- I/O backend (console, block device) via lockfree ring buffers in pinned memory

**Michael** then raised the stakes significantly:

> 64 bit and we are looking to at least boot the linux kernel and hopefully a full-fledged distro like ubuntu. Booting Linux means we need RV64GC (64-bit base + multiply + atomics + compressed + floating point), SV39 MMU, M/S/U privilege modes, PLIC interrupt controller, UART, and a proper device tree. Gotta collect boot stats but only when the debug env variable is set.

Michael uploaded the RISC-V Unprivileged and Privileged specification PDFs at this point.

**This is where the conversation stands — beginning implementation of a GPU-native RV64GC hypervisor targeting Linux boot.**

---

## Appendix A: Hardware

All benchmarks ran on:
- **GPU:** NVIDIA RTX PRO 6000 Blackwell Server Edition (188 SMs, 2430 MHz, 102.0 GB VRAM, 1598 GB/s bandwidth)
- **Platform:** Modal cloud compute

---

## Appendix B: Key Findings Summary

| Experiment | Result | Significance |
|---|---|---|
| Parallel VMs (50K) | 117x | Valid — embarrassingly parallel, but "just many tasks" |
| Parallel DFA (500M) | 248x (kernel), 18.7x (with transfer) | **Key result** — single serial computation restructured via algebraic properties |

The core insight: **any serial computation whose state updates form a monoid (associative operation with identity) can be parallelized via reduction.** The hardware parallelism exists; the challenge is restructuring algorithms to expose it.
