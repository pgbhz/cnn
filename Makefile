CXX := g++-15
CXXFLAGS := -O2 -std=c++17 -Wall -Wextra -pedantic -fopenmp
LDFLAGS  := -fopenmp

SRC_DIR := src
BIN_DIR := bin
TARGET := $(BIN_DIR)/cnn_minimal
SRC := $(SRC_DIR)/main.cpp

# Valor padrão de threads se não for passado
NUM_THREADS ?= 1

.PHONY: all clean run

all: $(TARGET)

$(BIN_DIR):
	@mkdir -p $(BIN_DIR)

$(TARGET): $(SRC) | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

run: $(TARGET)
	@echo "Executando com NUM_THREADS=$(NUM_THREADS)"
	@/usr/bin/time -l env OMP_NUM_THREADS=$(NUM_THREADS) OMP_DYNAMIC=0 ./$(TARGET)

clean:
	rm -rf $(BIN_DIR)
