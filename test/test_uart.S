    .section .text
    .globl _start
_start:
    # Write "GLAE\n" to UART at 0x10000000
    li      a0, 0x10000000
    li      a1, 'G'
    sb      a1, 0(a0)
    li      a1, 'L'
    sb      a1, 0(a0)
    li      a1, 'A'
    sb      a1, 0(a0)
    li      a1, 'E'
    sb      a1, 0(a0)
    li      a1, '\n'
    sb      a1, 0(a0)

    # SBI ecall test: putchar 'O' 'K' '\n' via legacy SBI
    li      a7, 1           # SBI_CONSOLE_PUTCHAR
    li      a0, 'O'
    ecall
    li      a7, 1
    li      a0, 'K'
    ecall
    li      a7, 1
    li      a0, '\n'
    ecall

    # Halt via SBI SRST
    li      a7, 0x53525354  # SRST extension
    li      a6, 0           # sbi_system_reset
    li      a0, 0           # SHUTDOWN
    li      a1, 0           # NO_REASON
    ecall

    # Fallback: infinite loop
1:  wfi
    j       1b
