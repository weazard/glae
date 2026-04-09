NVCC     := nvcc
NVFLAGS  := -arch=sm_90 -O2 -std=c++17 --expt-relaxed-constexpr -lineinfo
TARGET   := glae
SRC      := src/main.cu

.PHONY: all clean run run_debug test_payload

all: $(TARGET)

$(TARGET): $(SRC) $(wildcard src/**/*.cuh src/**/*.h)
	$(NVCC) $(NVFLAGS) -o $@ $<

test_payload: test/test_uart.bin

test/test_uart.bin: test/test_uart.S
	riscv64-linux-gnu-as -march=rv64gc -o test/test_uart.o $<
	riscv64-linux-gnu-ld -Ttext=0x80000000 -o test/test_uart.elf test/test_uart.o
	riscv64-linux-gnu-objcopy -O binary test/test_uart.elf $@
	rm -f test/test_uart.o test/test_uart.elf

run: $(TARGET) test_payload
	./$(TARGET) test/test_uart.bin

run_debug: $(TARGET) test_payload
	GLAE_DEBUG=1 ./$(TARGET) test/test_uart.bin

clean:
	rm -f $(TARGET) test/*.o test/*.elf test/*.bin
