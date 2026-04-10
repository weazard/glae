#pragma once
#include <cstdint>
#include <cstring>

// ============================================================
// FDT (Flattened Device Tree) binary generator — host code
// ============================================================

#define FDT_MAGIC       0xD00DFEED
#define FDT_BEGIN_NODE  0x00000001
#define FDT_END_NODE    0x00000002
#define FDT_PROP        0x00000003
#define FDT_END         0x00000009

// Big-endian helpers
static inline uint32_t be32(uint32_t v) {
    return __builtin_bswap32(v);
}
static inline uint64_t be64(uint64_t v) {
    return __builtin_bswap64(v);
}

struct FdtBuilder {
    uint8_t* buf;
    int struct_off;
    int strings_off;
    int strings_size;
    char strings[4096];

    void init(uint8_t* buffer) {
        buf = buffer;
        struct_off = 0;
        strings_off = 0;
        strings_size = 0;
        memset(strings, 0, sizeof(strings));
    }

    int add_string(const char* s) {
        // Check if string already exists
        for (int i = 0; i < strings_size; ) {
            if (strcmp(strings + i, s) == 0) return i;
            i += strlen(strings + i) + 1;
        }
        int off = strings_size;
        strcpy(strings + strings_size, s);
        strings_size += strlen(s) + 1;
        return off;
    }

    void put32(uint32_t v) {
        *(uint32_t*)(buf + struct_off) = be32(v);
        struct_off += 4;
    }

    void put_string(const char* s) {
        int len = strlen(s) + 1;
        memcpy(buf + struct_off, s, len);
        struct_off += (len + 3) & ~3; // pad to 4 bytes
    }

    void begin_node(const char* name) {
        put32(FDT_BEGIN_NODE);
        put_string(name);
    }

    void end_node() {
        put32(FDT_END_NODE);
    }

    void prop_u32(const char* name, uint32_t val) {
        put32(FDT_PROP);
        put32(4);
        put32(add_string(name));
        put32(val);
    }

    void prop_u64(const char* name, uint64_t val) {
        put32(FDT_PROP);
        put32(8);
        put32(add_string(name));
        *(uint32_t*)(buf + struct_off) = be32((uint32_t)(val >> 32));
        struct_off += 4;
        *(uint32_t*)(buf + struct_off) = be32((uint32_t)val);
        struct_off += 4;
    }

    void prop_str(const char* name, const char* val) {
        int len = strlen(val) + 1;
        put32(FDT_PROP);
        put32(len);
        put32(add_string(name));
        memcpy(buf + struct_off, val, len);
        struct_off += (len + 3) & ~3;
    }

    void prop_cells(const char* name, const uint32_t* cells, int n) {
        put32(FDT_PROP);
        put32(n * 4);
        put32(add_string(name));
        for (int i = 0; i < n; i++) {
            *(uint32_t*)(buf + struct_off) = be32(cells[i]);
            struct_off += 4;
        }
    }

    void prop_empty(const char* name) {
        put32(FDT_PROP);
        put32(0);
        put32(add_string(name));
    }

    void prop_phandle(const char* name, uint32_t ph) {
        prop_u32(name, ph);
    }
};

// Build the device tree for our virtual machine
static int build_fdt(uint8_t* buffer, uint64_t dram_base, uint64_t dram_size,
                     const char* bootargs, int num_harts = 1) {
    // Reserve space for header (40 bytes) + memory reservation (16+16 bytes)
    int header_size = 40;
    int memrsv_size = 16; // one empty entry (8+8 bytes of zeros)

    FdtBuilder fdt;
    fdt.init(buffer + header_size + memrsv_size);

    // Phandle assignments: cpu intc phandles = 10+i, plic = 10+num_harts
    // (start at 10 to avoid conflicts)
    uint32_t ph_plic = 10 + num_harts;

    // Root node
    fdt.begin_node("");
    fdt.prop_u32("#address-cells", 2);
    fdt.prop_u32("#size-cells", 2);
    fdt.prop_str("compatible", "riscv-virtio");
    fdt.prop_str("model", "glae,rv64");

    // /chosen
    fdt.begin_node("chosen");
    fdt.prop_str("bootargs", bootargs);
    fdt.prop_str("stdout-path", "/soc/serial@10000000");
    fdt.end_node();

    // /memory@80000000
    fdt.begin_node("memory@80000000");
    fdt.prop_str("device_type", "memory");
    uint32_t reg_mem[] = {
        (uint32_t)(dram_base >> 32), (uint32_t)dram_base,
        (uint32_t)(dram_size >> 32), (uint32_t)dram_size
    };
    fdt.prop_cells("reg", reg_mem, 4);
    fdt.end_node();

    // /cpus — generate one node per hart
    fdt.begin_node("cpus");
    fdt.prop_u32("#address-cells", 1);
    fdt.prop_u32("#size-cells", 0);
    fdt.prop_u32("timebase-frequency", 10000000);

    for (int h = 0; h < num_harts; h++) {
        char name[32];
        snprintf(name, sizeof(name), "cpu@%d", h);
        fdt.begin_node(name);
        fdt.prop_str("device_type", "cpu");
        fdt.prop_u32("reg", h);
        fdt.prop_str("compatible", "riscv");
        fdt.prop_str("riscv,isa", "rv64imafdcsu");
        fdt.prop_str("mmu-type", "riscv,sv39");
        fdt.prop_str("status", "okay");

        fdt.begin_node("interrupt-controller");
        fdt.prop_u32("#interrupt-cells", 1);
        fdt.prop_str("compatible", "riscv,cpu-intc");
        fdt.prop_empty("interrupt-controller");
        fdt.prop_phandle("phandle", 10 + h);
        fdt.end_node();

        fdt.end_node();
    }
    fdt.end_node(); // cpus

    // /soc
    fdt.begin_node("soc");
    fdt.prop_u32("#address-cells", 2);
    fdt.prop_u32("#size-cells", 2);
    fdt.prop_str("compatible", "simple-bus");
    fdt.prop_empty("ranges");

    // clint@2000000 — interrupts-extended lists all harts
    fdt.begin_node("clint@2000000");
    fdt.prop_str("compatible", "riscv,clint0");
    uint32_t reg_clint[] = { 0, 0x02000000, 0, 0x10000 };
    fdt.prop_cells("reg", reg_clint, 4);
    {
        // Each hart: (phandle, 3=MSI, phandle, 7=MTI)
        uint32_t clint_irqs[MAX_HARTS * 4];
        for (int h = 0; h < num_harts; h++) {
            clint_irqs[h*4+0] = 10 + h;  // cpu intc phandle
            clint_irqs[h*4+1] = 3;       // M-mode software interrupt
            clint_irqs[h*4+2] = 10 + h;
            clint_irqs[h*4+3] = 7;       // M-mode timer interrupt
        }
        fdt.prop_cells("interrupts-extended", clint_irqs, num_harts * 4);
    }
    fdt.end_node();

    // plic@c000000 — interrupts-extended lists all harts
    fdt.begin_node("plic@c000000");
    fdt.prop_str("compatible", "sifive,plic-1.0.0");
    uint32_t reg_plic[] = { 0, 0x0c000000, 0, 0x04000000 };
    fdt.prop_cells("reg", reg_plic, 4);
    fdt.prop_u32("#interrupt-cells", 1);
    fdt.prop_empty("interrupt-controller");
    {
        // Each hart: (phandle, 11=MEI, phandle, 9=SEI)
        // Order determines context IDs: ctx hart*2=M-mode, ctx hart*2+1=S-mode
        uint32_t plic_irqs[MAX_HARTS * 4];
        for (int h = 0; h < num_harts; h++) {
            plic_irqs[h*4+0] = 10 + h;
            plic_irqs[h*4+1] = 11;  // M-mode external interrupt (context h*2)
            plic_irqs[h*4+2] = 10 + h;
            plic_irqs[h*4+3] = 9;   // S-mode external interrupt (context h*2+1)
        }
        fdt.prop_cells("interrupts-extended", plic_irqs, num_harts * 4);
    }
    fdt.prop_u32("riscv,ndev", 64);
    fdt.prop_phandle("phandle", ph_plic);
    fdt.end_node();

    // serial@10000000
    fdt.begin_node("serial@10000000");
    fdt.prop_str("compatible", "ns16550a");
    uint32_t reg_uart[] = { 0, 0x10000000, 0, 0x100 };
    fdt.prop_cells("reg", reg_uart, 4);
    fdt.prop_u32("clock-frequency", 3686400);
    fdt.prop_phandle("interrupt-parent", ph_plic);
    uint32_t uart_irq[] = { 10 };
    fdt.prop_cells("interrupts", uart_irq, 1);
    fdt.end_node();

    fdt.end_node(); // soc
    fdt.end_node(); // root

    fdt.put32(FDT_END);

    int struct_size = fdt.struct_off;
    int strings_start = header_size + memrsv_size + struct_size;
    int total_size = strings_start + fdt.strings_size;
    total_size = (total_size + 7) & ~7; // align to 8

    // Copy strings
    memcpy(buffer + strings_start, fdt.strings, fdt.strings_size);

    // Memory reservation block (one empty entry)
    memset(buffer + header_size, 0, memrsv_size);

    // Write header
    uint32_t* hdr = (uint32_t*)buffer;
    hdr[0] = be32(FDT_MAGIC);
    hdr[1] = be32(total_size);
    hdr[2] = be32(header_size + memrsv_size);  // off_dt_struct
    hdr[3] = be32(strings_start);               // off_dt_strings
    hdr[4] = be32(header_size);                  // off_mem_rsvmap
    hdr[5] = be32(17);                           // version
    hdr[6] = be32(16);                           // last_comp_version
    hdr[7] = be32(0);                            // boot_cpuid_phys
    hdr[8] = be32(fdt.strings_size);             // size_dt_strings
    hdr[9] = be32(struct_size);                  // size_dt_struct

    return total_size;
}
