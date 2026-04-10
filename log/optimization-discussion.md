# GLAE Optimization Discussion — 2026-04-09

## User: Optimization Ideas

> i'll throw some ideas at a wall but they don't necessarily have merit. feel free to dismiss.

### Shared memory register file
The guest's x[32] is 256 bytes. Shared memory access is ~5 cycles vs ~30 for L1 hit, ~400 for L1 miss. At kernel start, copy hart->x into shared memory, execute the whole batch from there, copy back at end. With 32 harts per block, that's 8KB of shared memory for integer registers — well within the 228KB available per SM. Add f[32] for another 8KB.

### Decode table
Your switch-on-opcode-then-switch-on-funct3-then-switch-on-funct7 creates deeply nested branches. GPUs have no branch predictor — every branch is a pipeline flush. Replace with a flat function pointer table indexed by (opcode << 3) | funct3, falling back to a secondary table for funct7 disambiguation. One indirect jump instead of three nested conditionals.

### Warp-per-hart with shuffle communication
Instead of 1 thread per hart, keep the 32-thread warp but use it productively. Thread 0 executes. Threads 1-7 speculatively decode instructions at PC+2, PC+4, PC+6, etc. Thread 8-15 prefetch TLB entries for the next few memory addresses. Use __shfl_sync to pass results between lanes at register speed (no shared memory needed). When a branch is taken, threads 1-7 discard their work — but for straight-line code (which dominates), you get decode overlap for free.

### Coalesce page table walks
Your SV39 walk does 3 sequential global memory reads (one per level). For multiple harts on the same SM, many will be walking the same page tables (same kernel text, same data structures). If you batch TLB misses across harts in the same block and sort by PTE address, the memory reads coalesce — the GPU issues one memory transaction for 32 reads to the same cache line.

### Stimecmp-driven sleep instead of polling
Your timer check runs every 512 instructions:
```cuda
if ((i & 0x1FF) == 0) {
    clint_tick(hart, mach->clint);
```
This is ~100 instructions of timer logic every 512 guest instructions — ~20% overhead. Instead, precompute at batch start how many cycles until the next timer fires, and only check when that count is reached. For WFI harts, skip them entirely in the scheduler rather than launching them and immediately yielding.

### Cooperative TLB across harts
When hart A does a page table walk, the result is valid for any hart running the same process (same ASID/satp). Right now each hart has its own 256-entry TLB and walks independently. Instead, add a shared L2 TLB in shared memory — say 1024 entries per block. When hart A completes a walk, it inserts into both its private L1 TLB and the shared L2. When hart B misses its L1, it checks the shared L2 before walking. During Linux boot, all harts execute the same kernel code with the same page tables — the first hart to touch a page pays the 1200-cycle walk, every subsequent hart gets a 5-cycle shared memory hit.

### Stalled-thread work donation
When a thread stalls on a global memory load (LR/SC, AMO, page table walk), it can't do anything for its own hart. But it can do prefetch work for neighboring harts. The pattern:
```cuda
// Thread is about to stall on a memory load for its own hart
uint64_t my_addr = /* page table entry address */;
uint64_t neighbor_pc = neighbor_hart->pc;  // read from shared mem

// Issue both loads simultaneously — GPU coalesces them
uint64_t my_pte = *(uint64_t*)(dram + my_addr);
uint64_t neighbor_insn = *(uint32_t*)(dram + neighbor_pc);  // prefetch

// My load was the one I needed; neighbor's load warmed the cache
// for when that thread executes next
```
The stall costs 400 cycles either way. But by issuing a second load for a neighbor, the GPU's memory controller can sometimes coalesce or pipeline them, and the neighbor's next fetch hits L1 instead of DRAM.

### Shared decode cache
Linux harts running the same kernel execute the same instructions at the same addresses. Your per-hart icache duplicates work — all 4 harts independently decode the same compressed instruction at 0x80001066. Put a shared decode cache in shared memory (keyed by physical address, so it survives TLB differences). One hart decodes, all harts benefit. For kernel text this would be nearly 100% hit rate after warmup.

### Cooperative interrupt delivery
Right now your PLIC check runs on hart 0 only:
```cuda
if (hart_id == 0)
    plic_update_ext(hart, mach->plic);
```
Instead, distribute interrupt work: each hart checks its own pending bits, but one designated "monitor" thread per block scans all harts' mip and mie fields and pre-computes which harts have actionable interrupts. It writes a single flag per hart into shared memory. The per-hart interrupt check becomes a single shared memory read instead of the full priority scan.

### Adaptive batch sizing per hart
Currently every hart runs exactly 50,000 instructions per batch. But a hart executing a tight loop (like memset) doesn't need timer checks or PLIC scans — it'll run the full batch without interruption. A hart doing heavy I/O (SBI calls, UART writes) yields early anyway. Let harts communicate their "productivity" via shared memory:
```cuda
__shared__ uint32_t batch_progress[MAX_HARTS_PER_BLOCK];

// Every 1024 instructions, update progress
batch_progress[local_hart_id] = i;

// If I'm in WFI, check if any neighbor needs help
if (hart->wfi) {
    // Find the busiest hart and prefetch for it
    int busiest = find_max(batch_progress);
    prefetch_for(harts[busiest]);
}
```
WFI harts become prefetch engines for active harts instead of burning cycles polling for interrupts.

### Memory access reordering across harts
This is the most speculative idea but potentially highest impact. When multiple harts in the same warp all need global memory loads, the GPU issues them together. If you can align when harts do their memory accesses, you maximize coalescing. The trick: after decode, each thread announces "I need to read address X" via __shfl_sync. A lightweight sorter groups nearby addresses. Threads whose addresses fall in the same cache line get combined into one transaction. For kernel code pages, many harts will be fetching from the same or adjacent pages — coalescing could cut memory transactions by 4-8x.

### Epoch-based reclamation for TLB
Linux's RCU (read-copy-update) lets readers access shared data without locks — writers wait until all readers finish their current "epoch" before freeing old data. Apply this to TLB invalidation: when one hart writes satp or does SFENCE.VMA, don't immediately flush all harts' TLBs (which is expensive cross-SM communication). Instead, bump a global epoch counter. Each hart checks the epoch at batch boundaries and lazily flushes if stale. This is exactly how Linux handles TLB shootdowns on real hardware — the IPI just sets a flag, the target core flushes when it next enters the kernel.

### Run-to-completion event loop
Borrowed from DPDK/network stack design. Instead of the current "execute N instructions then yield" model, structure each hart as an event loop: execute until a natural yield point (ecall, WFI, page fault, timer), handle the event inline without returning to the CPU, then resume. The CPU host only gets involved for actual external I/O. This eliminates the kernel launch overhead (~5-10μs) for events that can be resolved GPU-side. Timer interrupts, IPI delivery, TLB flushes — all handled within the persistent kernel.

### Priority-based hart scheduling
Borrowed from OS scheduler design (CFS, priority queues). Not all harts are equally productive. A hart spinning on a spinlock is burning cycles without progress. A hart executing memcpy is doing useful work. Track per-hart IPC (instructions that modify architectural state vs total cycles) in shared memory. A lightweight scheduler thread within the block can deprioritize spinning harts — give them fewer iterations per batch, or skip them entirely for a few rounds. The freed SM cycles go to productive harts. This mirrors how hypervisors like KVM credit/debit vCPU scheduling based on whether the guest is halted.

### Speculative lock elision
Intel TSX tried this in hardware — execute a critical section optimistically without acquiring the lock, roll back if a conflict is detected. For GPU emulation: when a hart hits an amoswap (spinlock acquire), speculatively continue executing with the lock "held" but don't issue the actual atomicCAS to global memory. Record all stores to a thread-local write buffer in shared memory. If no other hart touches the same cache line before the lock release, commit the buffer atomically. If conflict detected, discard and retry with the real atomic. This avoids the ~400-cycle global memory atomic for uncontended locks, which is the common case in early boot.

### Coroutine-style context switching
Borrowed from green threads / goroutines. Each hart doesn't need a full GPU thread — it needs a program counter, registers, and the ability to yield. Implement M virtual harts multiplexed onto N GPU threads (M >> N) using cooperative multitasking. When a hart stalls on memory, it saves its state to shared memory and the physical thread picks up a different hart from a ready queue. This is essentially userspace threading (like Go's goroutine scheduler) but at the GPU level. You decouple "number of emulated cores" from "number of GPU threads," letting you run thousands of harts on a single SM.

### Warp-divergence-aware basic block fusion
SIMT executes all threads in a warp together. If 32 harts in a warp are all executing different instructions, the warp serializes. But during kernel boot, many harts execute identical code paths (same function, same basic block, different data). Detect when multiple harts within a warp have the same PC. When they do, fuse their execution: decode the instruction once, broadcast it via __shfl_sync, all harts execute the same opcode simultaneously with zero divergence. The warp achieves true SIMT parallelism — 32 harts executing at the speed of 1. For kernel init code where all secondaries run the same boot stub, this could give a literal 32x speedup per warp.

### Memory-access shadow execution
Run each hart on two threads: the "real" thread and a "shadow" thread one basic block ahead. The shadow thread executes a stripped-down version of the instruction stream — it only computes memory addresses (ignoring ALU results that don't feed into load/store addresses). It issues prefetch loads for every address it computes. The real thread, running one block behind, finds every memory access already in L1. The shadow doesn't need full register state — just the registers that feed into address calculations (typically sp, gp, a few pointer registers). On GPU, the shadow thread is free because it would otherwise be idle SIMT lanes. The shadow's incorrect speculative results are never committed — it exists purely to warm the cache.

### Thermodynamic instruction scheduling
Assign each hart a "temperature" based on its recent memory stall ratio. Hot harts (mostly stalling on memory) should be interleaved with cold harts (mostly ALU-bound) within the same warp. Periodically reshuffle hart-to-thread assignments so that each warp has a mix of hot and cold. When a hot hart stalls, the warp switches to a cold hart that can execute immediately. This maximizes the probability that at least one thread in the warp is ready to execute at any given cycle. The "temperature" metaphor maps to simulated annealing — the scheduler continuously adjusts assignments to minimize total warp stall time, with random perturbations to escape local minima.

### Differential state compression for context switches
When swapping hart state in the coroutine model above, you're copying ~2KB (registers + CSRs + TLB) to/from shared memory. But between context switches, most state doesn't change — a typical basic block touches 3-4 registers. Track a dirty bitmask (32 bits for x[], 32 for f[], 16 for CSRs) in a register. On context-out, only save dirty registers. On context-in, only restore what the next hart dirtied on its last run. With __popc and __ballot_sync you can compute the minimal save/restore set in one cycle. For a typical switch, you copy 3-4 registers (24-32 bytes) instead of 512, making virtual hart switching nearly free.

### Cross-SM instruction wavefront propagation
When hart 0 on SM 0 decodes an instruction at address X, that decoded result is useful for any hart on any SM that will execute address X. Use GPU global memory (L2-cached) as a system-wide decoded instruction broadcast channel. Each SM writes its decoded instructions to a global decode cache. Other SMs check this cache before decoding. L2 cache ensures the reads are fast (~100 cycles vs ~400 for DRAM). The propagation is eventually-consistent — a hart might decode redundantly if the cache hasn't propagated yet, but never incorrectly. During kernel boot, SM 0 decodes the boot path and every subsequent SM finds it pre-decoded. This turns the decode cost from O(N×M) (N harts × M instructions) to O(M) amortized — one decode per unique instruction across the entire system.

---

## Claude: Analysis and Tiering

### Tier 1: High impact, straightforward to implement

**Shared memory register file** — Yes. This is the single most impactful quick win. `__shared__ uint64_t sx[32]` per hart, copy in at batch start, execute from shared memory (~5 cycles), copy back at end. 256 bytes per hart. With the multi-hart-per-SM model you're building toward, 8 harts × 512B (x+f) = 4KB, trivially fits in 228KB shared. Expected: **15-30% MIPS improvement**.

**Decode table** — Yes. The nested switch generates a branch cascade on every instruction. A flat `__device__ (*handler[128])(HartState*, Machine*, uint32_t)` indexed by opcode eliminates ~3 branches per instruction. Expected: **5-15% MIPS improvement**.

**Stimecmp-driven sleep** — Yes. Precompute `min(stimecmp, mtimecmp) - current_mtime` at batch start, only check timer when `instret` exceeds that delta. Eliminates the every-512-instruction timer check overhead. Expected: **10-20% MIPS improvement**.

### Tier 2: High impact, requires architectural change

**Warp-per-hart with shuffle communication** — This is the multi-warp idea done right. The key insight is using `__shfl_sync` instead of shared memory for inter-lane communication. Thread 0 executes, threads 1-7 speculatively decode at PC+2/4/6..., threads 8-15 prefetch TLB. The risk is warp divergence killing the benefit when branches are frequent. But for straight-line code (memset loops, kernel init), it could give **1.5-2x**. Worth prototyping.

**Coroutine-style context switching** — This IS the multi-hart-per-SM idea. M virtual harts on N GPU threads with cooperative scheduling. When a hart stalls on memory, save 3-4 dirty registers (your differential compression idea), pick up another hart. This directly addresses the latency-hiding problem. With 4-8 virtual harts per warp, memory stalls overlap. Expected: **2-4x per SM**.

**Run-to-completion event loop** — Persistent kernel that handles timer/IPI/TLB inline without returning to host. This eliminates the 5-10µs kernel launch overhead per batch AND the WFI clock problem. The kernel just loops forever; the host only interrupts for UART I/O. Expected: **significant reduction in WFI idle waste**.

### Tier 3: Clever but complex, needs measurement

**Coalesced page table walks / Cooperative TLB** — Both solve the same problem: redundant page walks for shared kernel text. A shared L2 TLB in shared memory (1024 entries) would catch most kernel-text misses after the first walk. The challenge is indexing and eviction policy in shared memory. Expected: **20-40% fewer page walks**, which reduces memory stalls.

**Shared decode cache** — Similar to cooperative TLB but for the icache. A shared decode cache in shared memory keyed by physical address would give near-100% hit rate for kernel text across harts. Expected: **eliminates redundant decode work**, moderate impact since icache already helps.

**Warp-divergence-aware basic block fusion** — This is the SIMT dream: when N harts in a warp are at the same PC, execute once and broadcast. During secondary hart boot (identical code path), this literally gives Nx speedup. During normal kernel execution with different code paths, divergence dominates and the benefit drops. Expected: **2-32x for homogeneous workloads**, but complex to detect and exploit.

**Memory-access shadow execution** — Shadow thread for address prefetch. The shadow only computes addresses, not full results. Free because idle SIMT lanes. Expected: **meaningful L1 hit rate improvement**, but depends on address computation being predictable.

### Tier 4: Speculative, high complexity

**Speculative lock elision** — Interesting for uncontended locks (common in boot) but the rollback machinery is complex. Need a write buffer, conflict detection, and abort path. GPU shared memory makes the buffer cheap, but the tracking overhead might eat the savings.

**Thermodynamic scheduling** — Elegant metaphor but the reshuffle cost (changing thread assignments mid-kernel) is nontrivial. Simpler approaches (priority queues, round-robin between hot/cold harts) might capture 80% of the benefit.

**Cross-SM instruction wavefront** — The L2 cache already provides some of this naturally. An explicit broadcast adds complexity (cache coherence, eviction) with diminishing returns since L2 is already shared.

**Epoch-based TLB reclamation** — Linux already does this via SBI RFENCE. The current implementation flushes eagerly, but lazy invalidation with epoch counters is cleaner and avoids cross-SM communication. Low complexity, moderate benefit.

### Critical path to 10x+

The **critical path to 10x+ improvement** is:
1. Multi-hart-per-SM (coroutine model) → hides memory latency → **2-4x**
2. Shared memory registers → faster access → **1.3x**
3. Persistent kernel → eliminates launch overhead + WFI problem → **1.5x**
4. Decode table + timer optimization → reduces per-instruction overhead → **1.2x**

Compounded: ~6-10x, bringing you from 2.3 MIPS/hart to 15-23 MIPS/hart. At 188 SMs × 4 virtual harts × 15 MIPS = **~11 GIPS aggregate** theoretical.

---

## The WFI Timer Problem and Its Solution

### Problem
`clock64()` is a per-SM cycle counter that only ticks while GPU code is running. When a hart enters WFI, the kernel yields immediately, and `clock64()` stops advancing for that hart. Guest `mtime` freezes. Timer interrupts never fire. Idle harts never wake.

### Attempted Solutions

1. **`%globaltimer` (GPU global nanosecond timer)** — Ticks continuously and is synchronized across SMs. But it makes mtime advance at wall-clock speed while the emulator only executes ~2M insns/s. The guest sees 10M timer ticks per second but only completes 2M instructions — spending 80% of time servicing timer interrupts instead of doing init work. Rejected.

2. **Host-side `gpu_clock_base` adjustment (hybrid approach)** — Keep `clock64()` for proportional time during execution. When a hart is in WFI, the host subtracts `gpu_freq / 1000` from that hart's `gpu_clock_base` during each 1ms sleep cycle. This makes `get_mtime()` appear to advance by 1ms even though `clock64()` didn't move. Works but is a hack.

### The Real Fix: Persistent Kernel (Run-to-Completion)

The user's "run-to-completion event loop" idea solves this completely. With a persistent kernel that never exits:

```cuda
__global__ void vcpu_run_persistent(HartState* harts, Machine* mach) {
    HartState* hart = &harts[blockIdx.x];

    while (true) {  // never exits
        // ... execute instructions ...

        if (hart->wfi) {
            // WFI = spin inside the kernel. clock64() keeps ticking.
            while (!(hart->mip & hart->mie)) {
                clint_tick(hart, mach->clint);
                if (hart->stimecmp != 0 && hart->get_mtime() >= hart->stimecmp)
                    hart->mip |= MIP_STIP;
                __nanosleep(100);  // yield SM resources to other warps
            }
            hart->wfi = 0;
            continue;
        }

        // Yield to host only for external I/O (UART flush)
        if (hart->yield_reason == YIELD_UART_TX) {
            // signal host via pinned memory flag, spin until host clears it
        }
    }
}
```

`clock64()` never stops because the kernel never exits. WFI becomes a spin-wait that naturally advances `mtime`. Timer fires, hart wakes, zero host involvement.

This also eliminates the 5-10µs launch overhead per batch and the cudaDeviceSynchronize stall. The `__nanosleep` during WFI spin yields SM resources so other warps on the same SM can execute — directly enabling multi-hart-per-SM.
