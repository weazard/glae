# GLAE Progress Report

GPU-native RV64GC hypervisor — current state, challenges, and improvement strategies.

## Project Overview

GLAE is a RISC-V RV64GC hypervisor written entirely in CUDA, running guest
instructions on GPU shader cores. The project explores whether GPU parallelism
can be applied to traditionally serial computation (CPU emulation) by
redesigning the execution philosophy around GPU hardware.

**Hardware:** NVIDIA RTX PRO 6000 Blackwell Server Edition
- 188 Streaming Multiprocessors (SMs)
- 2430 MHz shader clock
- 98 GB VRAM, 1598 GB/s memory bandwidth
- 96 MB L2 cache, 128 KB L1 cache per SM
- 228 KB shared memory per SM

**Codebase:** ~4,400 lines across 25 files (23 CUDA headers, 1 CUDA source, 1 test).

---

## What Works

### Bare-Metal Execution (Phase 1)
A standalone test payload (`test/test_uart.S`) prints "GLAE" via direct UART
MMIO writes and "OK" via SBI legacy console putchar, then halts cleanly via
SBI SRST. This validates:
- RV64I base integer ISA (LUI, ADDI, SB, JAL, ECALL)
- C extension decompression (16-bit to 32-bit)
- Memory bus dispatch (DRAM and UART MMIO)
- 16550A UART emulation with pinned-memory ring buffers
- SBI ecall interception (console putchar, system reset)
- GPU kernel launch/yield/relaunch cycle

### Linux Boot (Phase 2+)
Linux 7.0-rc7 boots through the following stages:

```
[    0.000000] Linux version 7.0.0-rc7 (riscv64-linux-gnu-gcc 13.3.0)
[    0.000000] Machine model: glae,rv64
[    0.000000] SBI specification v0.2 detected
[    0.000000] SBI implementation ID=0x676c6165 Version=0x1
[    0.000000] earlycon: uart8250 at MMIO 0x0000000010000000 (options '115200')
[    0.000000] printk: legacy bootconsole [uart8250] enabled
[    0.000000] riscv: base ISA extensions acdfim
[    0.000000] Built 1 zonelists, mobility grouping on. Total pages: 32768
[    0.000000] SLUB: HWalign=64, Order=0-3, MinObjects=0, CPUs=1, Nodes=1
[    0.000392] sched_clock: 64 bits at 10MHz, resolution 100ns
[    0.107735] Calibrating delay loop (skipped) .. 20.00 BogoMIPS
[    1.833607] Memory: 102140K/131072K available (10067K kernel code)
[    2.052987] devtmpfs: initialized
[    4.343987] NET: Registered PF_NETLINK/PF_ROUTE protocol family
[    9.095263] raid6: using algorithm int64x2 gen() 1 MB/s
[   13.560924] clocksource: Switched to clocksource riscv_clocksource
[   23.148688] hrtimer: interrupt took 8719300 ns
```

This proves the following subsystems are fully functional:
- **SV39 MMU** with 3-level page table walk and software TLB (256 entries)
- **M/S/U privilege modes** with full trap delegation via medeleg/mideleg
- **CSR read/write** with privilege checking and sstatus/sip/sie aliasing
- **Trap handling** (entry, delegation, MRET, SRET)
- **SBI** (timer, console, base probes, remote fence, system reset)
- **CLINT** timer (mtime derived from GPU `clock64()`, scaled to 10 MHz)
- **PLIC** interrupt controller
- **FDT** device tree generation (CPU, memory, UART, CLINT, PLIC)
- **Timer interrupts** (scheduler ticks running at ~24 Hz)
- **M extension** (multiply/divide, including MULH via `__umul64hi`)
- **A extension** (LR/SC, AMO operations for spinlocks)
- **F/D extensions** (floating-point using native CUDA float/double)
- **C extension** (compressed instruction decompression)

---

## Current Performance

| Metric | Value |
|--------|-------|
| Emulation rate (1 hart) | 2.3 MIPS |
| Emulation rate (4 harts SMP) | 4.4 MIPS aggregate (scaling as secondaries boot) |
| GPU cycles per guest instruction | ~1,050 |
| GPU utilization (4 harts) | ~2% (4 of 188 SMs) |
| Batch size | 50,000 instructions per kernel launch |
| Kernel launch overhead | ~5-10 us per launch |
| Boot progress in 90s (4 harts) | ~230M instructions, kernel reaches SMP bringup |
| SMP status | Secondary harts started via SBI HSM, executing init code |

### SMP Boot Evidence

```
[    0.000000] SBI HSM extension detected
[    0.000000] SLUB: CPUs=4, Nodes=1
[    0.000000] rcu: RCU restricting CPUs from NR_CPUS=256 to nr_cpu_ids=4.
[    4.476483] smp: Bringing up secondary CPUs ...
[HSM] hart0 starting hart1 at 80001066
[HSM] hart0 starting hart2 at 80001066
[HSM] hart0 starting hart3 at 80001066
```

### Comparison to Established Emulators

| Emulator | MIPS | Technique |
|----------|------|-----------|
| QEMU TCG (x86 host) | 100-300 per hart | JIT compilation to native x86 |
| GLAE (1 hart, interpreted) | 2.3 | Serial GPU thread |
| GLAE (4 harts, SMP) | 4.4 aggregate | 4 SMs active, scaling with boot |
| GLAE (188 harts, projected) | ~430 aggregate | All SMs active |
| GLAE (188 harts + JIT, projected) | ~2,000-9,000 | JIT + all SMs |

---

## Challenges Encountered

### 1. Misaligned Memory Access (Critical, Fixed)
**Problem:** RISC-V C extension produces instructions at 2-byte-aligned addresses.
The instruction fetch code used `*(uint32_t*)(dram + offset)` which is undefined
behavior for addresses not aligned to 4 bytes. On GPU, this caused silent data
corruption — the GPU kernel would hang or produce wrong results.

**Fix:** Replaced all pointer-cast memory reads with `memcpy()`, which handles
arbitrary alignment correctly. This fixed both instruction fetch and page table
entry reads during SV39 walks.

### 2. Batch Size Hang (Critical, Fixed)
**Problem:** With `BATCH_SIZE=50000`, the GPU kernel would never return from
`cudaDeviceSynchronize()`. With `BATCH_SIZE=10000`, it worked fine.

**Root cause:** Without the instruction cache, hot loops (like kernel `memset`
or `memcpy`) fetched every instruction from DRAM on every iteration. At ~400
GPU cycles per DRAM read and 50,000 iterations, the GPU kernel ran for several
seconds — exceeding the GPU's kernel execution timeout.

**Fix:** The instruction cache (2048 entries) eliminated redundant DRAM reads for
hot loops. With cached instructions, 50,000 iterations complete in ~50ms.

### 3. UART Yield Storm (Performance, Fixed)
**Problem:** Every UART character write triggered `yield_reason = YIELD_UART_TX`,
causing the GPU kernel to exit, `cudaDeviceSynchronize()`, drain 1 character,
relaunch. During `printk` output (hundreds of characters), this added ~15us
overhead per character.

**Fix:** Characters accumulate in the ring buffer (4096 bytes). The kernel only
yields when the ring is 75% full. The host drains the ring after every batch
regardless.

### 4. Missing SBI Console Driver (Configuration, Fixed)
**Problem:** The Linux kernel built with `defconfig` did not include
`CONFIG_RISCV_SBI` or `CONFIG_SERIAL_EARLYCON_RISCV_SBI`. Boot arguments
`earlycon=sbi` produced no console output despite the SBI handler working.

**Fix:** Switched to `earlycon=uart8250,mmio,0x10000000,115200` which uses the
ns16550a UART driver directly. The kernel already had `CONFIG_SERIAL_8250=y`.

### 5. Inter-Warp Synchronization (Architecture, Unresolved)
**Problem:** Attempted multi-warp cooperative execution (8 warps: 1 execution +
7 prefetch helpers) using shared memory flags for coordination. Helper warps
failed to observe `running = 0` written by the main warp, causing the GPU
kernel to hang indefinitely.

**Root cause:** Complex interaction between `volatile` shared memory semantics,
CUDA's independent thread scheduling, and early thread return (`if (lane != 0) return`).
Shared memory writes from one warp are not guaranteed visible to other warps without
explicit fence instructions, and `__syncthreads()` after some threads have returned
causes undefined behavior.

**Status:** Reverted to single-warp execution. Multi-warp requires either
CUDA cooperative groups for proper inter-warp sync, or an architecture where
each warp is fully independent (SMP model).

### 6. Single-Thread Memory Latency (Fundamental)
**Problem:** At ~2,200 GPU clock cycles per guest instruction, the emulator runs
at 1.1 MIPS — far below the GPU's computational capability. The bottleneck is
not ALU throughput but memory access latency.

**Analysis:** Each guest instruction requires ~7 global memory accesses
(instruction fetch, register reads, register write, PC update). GPU global memory
latency is ~400 cycles per access with L1 miss, ~30 cycles with L1 hit. The
GPU hides this latency by scheduling other warps when one stalls, but with only
1 active warp, there is nothing to schedule.

A CPU handles the same workload faster because:
- Branch prediction (GPU has none — every branch stalls the pipeline)
- Out-of-order execution (GPU is strictly in-order per thread)
- Large per-core L1 cache optimized for single-thread access patterns
- Deep pipeline with speculative execution

**Status:** Fundamental architectural mismatch. Single-hart GPU emulation cannot
compete with CPU single-thread performance. The GPU advantage lies in parallelism.

---

## Improvement Strategies

### Single-Thread Optimizations (Diminishing Returns)

These improve per-hart MIPS but don't address the fundamental single-thread bottleneck.

#### Implemented
- **Instruction cache** (2048 entries): Caches decoded 32-bit instructions keyed
  by guest PC. Eliminates DRAM fetch for hot loops. Flushed on `SFENCE.VMA`
  and `satp` writes. Impact: fixed batch=50000 hang, marginal MIPS improvement.

- **UART batching**: Accumulate TX characters, yield only when ring buffer is
  75% full. Impact: eliminated hundreds of unnecessary kernel relaunches during
  boot.

- **Compiler optimization** (`-O3 --use_fast_math`): Enables aggressive inlining,
  fast math intrinsics. `__launch_bounds__(32, 1)` tells the compiler to optimize
  register allocation for a single warp. Impact: negligible MIPS improvement.

#### Potential (Not Yet Implemented)
- **Shared memory register file**: Copy `hart->x[32]` (256 bytes) to shared memory
  at kernel start, use throughout, copy back at end. Shared memory access is ~20
  cycles vs ~30+ for L1-cached global memory. Requires refactoring all execute
  functions to read/write from shared memory instead of the HartState pointer.
  Expected impact: 10-30% MIPS improvement.

- **Decode table**: Replace nested switch statements (opcode -> funct3 -> funct7)
  with a flat lookup table indexed by a pre-computed opcode hash. Eliminates
  computed jumps which are slow on GPU (no branch prediction). Expected impact:
  5-15% improvement.

- **Basic block cache**: Cache decoded instruction sequences for entire basic blocks
  (straight-line code between branches). Pre-compute register dependency wavefronts
  so independent instructions within a block can be identified. On re-execution,
  skip decode entirely and execute the pre-decoded bundle. Expected impact:
  20-40% improvement for loop-heavy code.

- **JIT compilation**: Translate hot basic blocks from RISC-V to native GPU
  instructions (PTX). The CPU-side JIT compiler would use `cuModuleLoadData` to
  compile and load optimized code. Eliminates all decode overhead and enables
  GPU-native optimization within blocks. Expected impact: 5-20x per-hart
  improvement, but complex to implement.

### Warp-Cooperative Optimizations (Medium Impact)

These use all 32 threads in a warp cooperatively for a single hart.

#### Attempted
- **SIMT parallel prefetch**: All 32 threads read from different addresses near
  the current PC, using SIMT to issue 32 memory reads simultaneously. Thread 0
  uses its read for the actual fetch; threads 1-31 warm the L1 cache.

  **Result:** Functional for the bare-metal test but caused issues with Linux boot
  due to synchronization complexity. The SIMT divergence overhead (threads 1-31
  idle during thread 0's execute phase) may negate the prefetch benefit.

- **Multi-warp helpers**: 8 warps — main execution warp + 7 prefetch warps that
  read DRAM near the current PC to warm L1/L2.

  **Result:** Failed due to inter-warp synchronization issues (see Challenge #5).

#### Potential (Not Yet Implemented)
- **Warp-cooperative pipeline**: Assign threads to specialized roles:
  - Threads 0-3: Execute current instruction and next independent ones
  - Threads 4-7: Pre-decode upcoming instructions
  - Threads 8-11: Pre-walk TLB for upcoming memory addresses
  - Threads 12-15: Prefetch instruction bytes from DRAM

  Communication via `__shfl_sync` (register-speed, no shared memory needed).
  The key challenge is SIMT divergence — different roles mean different code paths,
  which the GPU serializes within a warp. Independent thread scheduling (Volta+)
  helps but doesn't eliminate SIMT overhead.

  Expected impact: 2-4x improvement if divergence overhead is managed.

- **Wavefront-parallel execution**: Pre-analyze basic blocks for instruction-level
  parallelism. Group instructions into wavefronts based on register dependencies.
  Execute all instructions in a wavefront simultaneously using different warp
  threads. Average RISC-V basic block has ILP ~2-3, so a wavefront-parallel
  approach could execute 8 instructions in ~3 steps instead of 8.

  Expected impact: 1.5-2.5x for compute-heavy code. Limited by average basic
  block size (~6-8 instructions) and ILP (~2-3).

### GPU-Parallel Strategies (Highest Impact)

These are the approaches that actually utilize the GPU's massive parallelism.

#### SMP (Symmetric Multi-Processing)
Run N independent harts, one per SM. Each SM executes its own instruction stream
in parallel. Linux supports SMP out of the box — we already have all the per-hart
infrastructure (HartState, CSRs, TLB, trap handling).

**Implementation requires:**
1. Allocate N HartState structs (one per hart)
2. FDT with N CPU nodes
3. Inter-hart communication: IPI via SBI `sbi_send_ipi` setting SSIP on target hart
4. Shared memory: Guest DRAM is shared (atomics for coherence)
5. Hart boot protocol: Hart 0 boots first, others wait in HSM `STOPPED` state,
   Linux wakes them via `sbi_hart_start`
6. Kernel launch: `<<<N, 32>>>` — one block per SM, one warp per hart

**Expected performance:**
- 188 harts x 1.1 MIPS = ~207 MIPS aggregate
- True GPU utilization: ~100% of SMs active
- Linux SMP with 188 cores — massive parallel throughput

**Challenges:**
- Memory coherence: Multiple harts writing to shared DRAM requires careful
  handling of atomics (A extension) and cache coherence
- Interrupt routing: PLIC must route interrupts to the correct hart
- Timer per hart: Each hart needs its own mtimecmp
- Boot protocol: SBI HSM extension for hart lifecycle management

#### Bulk Operation Detection
Detect when the guest kernel performs bulk memory operations (memset, memcpy)
and implement them as GPU-parallel operations. Instead of emulating 1M
individual store instructions, issue a single `cudaMemset` or `cudaMemcpy`.

**Implementation:** Pattern-match store loops at the instruction level:
```
loop: sd  x0, 0(a0)   ; a0 = destination
      addi a0, a0, 8   ; advance pointer
      bne  a0, a1, loop ; until end
```
When detected, compute the range [a0, a1) and call `cudaMemsetAsync`.

**Expected impact:** Orders of magnitude faster for kernel init (which spends
significant time zeroing pages).

#### Device Emulation on Separate SMs
Run UART, PLIC, CLINT, and future virtio devices as independent persistent
kernels on dedicated SMs. Communication with the vCPU via lockfree ring
buffers in global memory.

**Benefit:** Decouples device emulation latency from the instruction execution
critical path. The vCPU never stalls on device I/O — it writes to a ring buffer
and continues.

---

## Implementation History

| Commit | Description | Key Change |
|--------|-------------|------------|
| `9ad5023` | Phase 1: Serial interpreter | Full RV64GC ISA, all devices, FDT, SBI |
| `3768d9c` | Linux boots on GPU | Fixed misaligned memcpy, 8250 earlycon |
| `f64672a` | Phase 2: Instruction cache | icache, UART batching, -O3 |
| `953e0d4` | Phase 3: Warp launch | 32-thread warp, slimmed kernel |
| `21cbb0b` | SMP infrastructure | Per-hart icache/CLINT/PLIC, HSM, IPI, N-cpu FDT |
| `24ab97d` | Zba/Zbb/Zbs extensions | Bit manipulation ISA for modern kernels |
| `(next)` | SMP boot verified | 4 harts booting, secondary CPUs starting, 4.4 MIPS |

---

## Glossary

**CUDA**
Compute Unified Device Architecture. NVIDIA's parallel computing platform and
API for running general-purpose code on GPU hardware. GLAE is written entirely
in CUDA C++.

**GPU (Graphics Processing Unit)**
A processor with thousands of simple cores designed for parallel workloads.
Unlike a CPU (which has few powerful cores optimized for serial speed), a GPU
trades single-thread performance for massive throughput. The RTX PRO 6000 has
188 SMs, each capable of running multiple warps simultaneously.

**SM (Streaming Multiprocessor)**
The fundamental processing unit of an NVIDIA GPU. Each SM contains ALUs, shared
memory, an instruction cache, and a warp scheduler. The RTX PRO 6000 Blackwell
has 188 SMs. In GLAE, each SM could run one independent hart (vCPU).

**Warp**
A group of 32 GPU threads that execute in lockstep (SIMT). All threads in a warp
execute the same instruction at the same time, but on different data. When threads
diverge (take different branches), the warp serializes both paths. In GLAE, one
warp runs one hart's instruction loop.

**SIMT (Single Instruction, Multiple Threads)**
The GPU execution model where all threads in a warp execute the same instruction
simultaneously. Analogous to SIMD (Single Instruction, Multiple Data) on CPUs,
but at the thread level. When GLAE issues a memory load, all 32 threads in the
warp load from different addresses in parallel — but only thread 0's result is
used for emulation.

**Hart**
RISC-V term for a hardware thread — an independent execution context with its
own PC, registers, CSRs, and privilege state. Equivalent to a CPU core in
x86 terminology. In GLAE, each hart is represented by a `HartState` struct
and runs on one GPU warp. SMP means multiple harts sharing memory.

**SMP (Symmetric Multi-Processing)**
A system with multiple harts (cores) sharing the same physical memory. Linux SMP
uses multiple harts for parallel execution. This is GLAE's primary path to GPU
utilization: run 188 harts (one per SM) for 188x aggregate throughput.

**MIPS (Million Instructions Per Second)**
Emulation throughput — the number of guest RISC-V instructions executed per
second. GLAE currently achieves 1.1 MIPS per hart. Not to be confused with the
MIPS processor architecture.

**RV64GC**
The RISC-V 64-bit ISA with extensions: G = General purpose (I+M+A+F+D), C =
Compressed instructions. This is the standard ISA target for Linux on RISC-V.
- **I**: Base integer (ALU, branches, loads/stores)
- **M**: Multiply/divide
- **A**: Atomics (LR/SC, AMO operations)
- **F**: Single-precision floating point
- **D**: Double-precision floating point
- **C**: Compressed 16-bit instruction encoding

**SV39**
RISC-V's 39-bit virtual addressing scheme. Uses a 3-level page table with 512
entries per level, supporting 4 KB pages, 2 MB megapages, and 1 GB gigapages.
GLAE implements the full SV39 walk with a 256-entry software TLB.

**CSR (Control and Status Register)**
Special RISC-V registers that control privilege mode, interrupts, virtual memory,
and performance counters. GLAE implements ~50 CSRs across M/S/U privilege levels.
Key CSRs: `mstatus` (global state), `satp` (page table pointer), `mtvec`/`stvec`
(trap vectors), `mip`/`mie` (interrupt pending/enable).

**SBI (Supervisor Binary Interface)**
The firmware interface between the OS kernel (S-mode) and the machine firmware
(M-mode). In GLAE, SBI calls are intercepted at the emulator level — there is
no actual M-mode firmware. The kernel calls `ecall` in S-mode, and GLAE handles
it directly (timer setup, console I/O, system reset, remote fence).

**TLB (Translation Lookaside Buffer)**
A cache of recent virtual-to-physical address translations. Without a TLB, every
memory access requires a 3-level page table walk (3 sequential DRAM reads at
~400 cycles each = ~1200 cycles). GLAE's TLB has 256 entries (separate I-TLB
and D-TLB) with direct-mapped indexing.

**PLIC (Platform-Level Interrupt Controller)**
Routes external device interrupts (e.g., UART) to specific harts. Each interrupt
source has a priority; each hart context has an enable mask and threshold. GLAE
emulates a 64-source PLIC with 2 contexts (M-mode and S-mode).

**CLINT (Core Local Interruptor)**
Provides per-hart timer and software interrupt functionality. GLAE derives
`mtime` from the GPU's `clock64()` intrinsic, scaled to match the RISC-V
10 MHz timebase frequency. `mtimecmp` triggers timer interrupts for the
scheduler.

**FDT (Flattened Device Tree)**
A binary data structure describing the hardware platform to the Linux kernel.
GLAE generates the FDT programmatically at boot, describing: CPU (rv64imafdcsu),
128 MB RAM, UART at 0x10000000, CLINT at 0x02000000, PLIC at 0x0C000000.

**Pinned Memory**
Host (CPU) memory allocated with `cudaMallocHost` that is page-locked and
directly accessible from both CPU and GPU without explicit copying. GLAE uses
pinned memory for the UART ring buffers — the GPU writes console output bytes
that the CPU reads and prints.

**Instruction Cache (icache)**
A software cache of recently fetched and decoded instructions, keyed by guest
PC. On a hit, the emulator skips the DRAM read and C-extension decompression,
using the cached 32-bit instruction directly. Flushed on `SFENCE.VMA` (code
modification) and `satp` writes (page table switch).

**Basic Block**
A sequence of instructions with no branches except at the end. The entry point
is the only way in; the exit (branch/jump) is the only way out. Basic blocks
are the natural unit for caching, analysis, and optimization — all instructions
in a block execute every time the block is entered.

**Batch Size**
The number of guest instructions the GPU kernel executes before returning control
to the CPU host. Larger batches amortize kernel launch overhead (~5-10 us) but
increase latency for I/O and timer updates. Currently 50,000 instructions per
batch.

**Yield**
When the GPU kernel exits mid-batch to return control to the CPU. Triggered by
UART buffer fullness, WFI (wait for interrupt), system halt, or batch completion.
The CPU services I/O, checks timers, and relaunches the GPU kernel.
