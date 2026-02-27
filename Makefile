OPENBLAS_PATH = /mnt/c/Users/marak/Desktop/TRPO/2_lab/openblas_test/OpenBLAS/install
CC = gcc
CFLAGS = -Wall -Wextra -O2 -g -I$(OPENBLAS_PATH)/include
LDFLAGS = -L$(OPENBLAS_PATH)/lib -lopenblas -lm

TARGET = test

all: $(TARGET)

$(TARGET): test.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

run1:
	@echo "=== ЗАПУСК С 1 ПОТОКОМ ==="
	LD_LIBRARY_PATH=$(OPENBLAS_PATH)/lib:$$LD_LIBRARY_PATH \
	OPENBLAS_NUM_THREADS=1 ./$(TARGET)

run2:
	@echo "=== ЗАПУСК С 2 ПОТОКАМИ ==="
	LD_LIBRARY_PATH=$(OPENBLAS_PATH)/lib:$$LD_LIBRARY_PATH \
	OPENBLAS_NUM_THREADS=2 ./$(TARGET)

run4:
	@echo "=== ЗАПУСК С 4 ПОТОКАМИ ==="
	LD_LIBRARY_PATH=$(OPENBLAS_PATH)/lib:$$LD_LIBRARY_PATH \
	OPENBLAS_NUM_THREADS=4 ./$(TARGET)

clean:
	rm -f $(TARGET)

.PHONY: all run1 run2 run4 clean