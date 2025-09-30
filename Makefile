CXX := g++
CXXFLAGS := -O2 -std=c++17 -Wall -Wextra -pedantic
LDFLAGS := 

SRC_DIR := src
BIN_DIR := bin
TARGET := $(BIN_DIR)/cnn_minimal
SRC := $(SRC_DIR)/main.cpp

.PHONY: all clean run

all: $(TARGET)

$(BIN_DIR):
	@mkdir -p $(BIN_DIR)

$(TARGET): $(SRC) | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

run: $(TARGET)
	$(TARGET)

clean:
	rm -rf $(BIN_DIR)
