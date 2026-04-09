# GLAE Usage Guide

## Building

```bash
# Requirements: CUDA toolkit (nvcc), RISC-V cross-toolchain
make
```

## Running

```bash
# Single hart (default)
./glae <kernel-image>

# SMP with N harts
./glae <kernel-image> --smp 4

# Custom DRAM size (default 128 MB)
./glae <kernel-image> --smp 4 --dram-mb 256

# Debug mode (SBI trace, periodic MIPS reporting)
GLAE_DEBUG=1 ./glae <kernel-image> --smp 4
```

## Building a Linux Kernel

```bash
git clone --depth 1 --branch v7.0-rc7 https://github.com/torvalds/linux.git /tmp/linux
cd /tmp/linux

# Configure for RISC-V SMP
make ARCH=riscv CROSS_COMPILE=riscv64-linux-gnu- defconfig
scripts/config --enable SMP --set-val NR_CPUS 256 \
  --disable MODULES --disable NETWORK --disable NETDEVICES \
  --disable USB_SUPPORT --disable SOUND --disable DRM --disable FB \
  --disable INPUT --disable HID --disable IOMMU_SUPPORT \
  --disable ACPI --disable PNP --disable EFI \
  --disable MEDIA_SUPPORT --disable THERMAL \
  --enable SERIAL_8250 --enable SERIAL_8250_CONSOLE \
  --enable SERIAL_OF_PLATFORM

make ARCH=riscv CROSS_COMPILE=riscv64-linux-gnu- olddefconfig
make ARCH=riscv CROSS_COMPILE=riscv64-linux-gnu- Image -j$(nproc)

cp arch/riscv/boot/Image /path/to/glae/Image
```

## Running the Bare-Metal Test

```bash
# Build test payload (requires riscv64-linux-gnu-as)
make test_payload

# Run test
./glae test/test_uart.bin
# Expected output: GLAE\nOK\n
```

## Debug Output

With `GLAE_DEBUG=1`, the emulator prints:
- SBI call trace (extension/function IDs, arguments)
- HSM hart lifecycle events (start/stop)
- Periodic aggregate stats: batch count, total instructions, MIPS, wall time
- Per-hart instruction counts at exit

## CLI Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--smp N` | 1 | Number of RISC-V harts (cores). Each hart runs on a separate GPU SM. |
| `--dram-mb N` | 128 | Guest DRAM size in megabytes. |
| `GLAE_DEBUG=1` | off | Enable debug output (SBI trace, MIPS stats). |

## Architecture

Each hart runs as one CUDA block (32 threads, 1 warp) on a separate GPU SM.
The kernel is launched as `<<<num_harts, 32>>>`. Guest DRAM is shared across
all harts in GPU global memory. UART I/O uses pinned-memory ring buffers for
zero-copy GPU-to-CPU communication.

## Supported ISA

RV64GC + Zba + Zbb + Zbs + Zicboz + Zihintpause

- **RV64I**: Base integer (all instructions)
- **M**: Multiply/divide
- **A**: Atomics (LR/SC, all AMO operations)
- **F**: Single-precision floating point
- **D**: Double-precision floating point
- **C**: Compressed 16-bit instructions
- **Zba**: Address generation (SH1ADD, etc.)
- **Zbb**: Basic bit manipulation (CLZ, CTZ, ROL, etc.)
- **Zbs**: Single-bit operations (BSET, BCLR, etc.)
- **Zicboz**: Cache block zero
- **Zihintpause**: PAUSE hint

## Known Limitations

- No V (vector) extension — kernel must be built without `CONFIG_RISCV_ISA_V`
- No block device — kernel boots to initramfs or hangs at rootfs mount
- No network — kernel must be built without `CONFIG_NETWORK`
- UART input is functional but untested with interactive shells
- Timer resolution is limited by GPU clock scaling (~100ns)
