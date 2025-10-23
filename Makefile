# =========================
# Compiladores e flags
# =========================
# OMP puro
CXX_OMP    := g++-15

# MPI puro (use o wrapper do MPI)
CXX_MPI    := mpicxx

# Híbrido (MPI + OpenMP via g++-15, linkando libmpi do Homebrew)
CXX_HYB    := g++-15

CXXFLAGS_OMP := -O2 -std=c++17 -Wall -Wextra -pedantic -fopenmp
LDFLAGS_OMP  := -fopenmp

CXXFLAGS_MPI := -O2 -std=c++17 -Wall -Wextra -pedantic
LDFLAGS_MPI  :=

# Descobre paths do OpenMPI instalado pelo Homebrew (macOS)
BREW_PREFIX := $(shell brew --prefix 2>/dev/null)
MPI_PREFIX  := $(shell brew --prefix open-mpi 2>/dev/null)
MPI_INC     := $(MPI_PREFIX)/include
MPI_LIB     := $(MPI_PREFIX)/lib

# Híbrido precisa dos headers/libs do MPI + OpenMP
CXXFLAGS_HYB := -O2 -std=c++17 -Wall -Wextra -pedantic -fopenmp -I$(MPI_INC)
# -Wl,-rpath para achar libmpi em tempo de execução sem setar DYLD_LIBRARY_PATH
LDFLAGS_HYB  := -fopenmp -L$(MPI_LIB) -Wl,-rpath,$(MPI_LIB) -lmpi

# =========================
# Estrutura de pastas
# =========================
SRC_DIR := src
BIN_DIR := bin

# Fontes / binários
OMP_SRC ?= $(SRC_DIR)/mainOMP.cpp
MPI_SRC ?= $(SRC_DIR)/mainMPI.cpp
HYB_SRC ?= $(SRC_DIR)/mainOMPMPI.cpp

OMP_TARGET := $(BIN_DIR)/cnn_minimal
MPI_TARGET := $(BIN_DIR)/cnn_mpi
HYB_TARGET := $(BIN_DIR)/cnn_hibrido

# =========================
# Parâmetros de execução
# =========================
NUM_THREADS ?= 1
PROCS       ?= 1
REPS        ?= 5

THREADS_LIST ?= 1 2 4 8
PROCS_LIST   ?= 1 2 4 8

.PHONY: all clean \
        omp mpi hybrid \
        run run_omp run_mpi run_hybrid \
        bench bench_omp bench_mpi bench_hybrid

# =========================
# Alvos principais
# =========================
all: $(OMP_TARGET) $(MPI_TARGET) $(HYB_TARGET)

omp: $(OMP_TARGET)
mpi: $(MPI_TARGET)
hybrid: $(HYB_TARGET)

$(BIN_DIR):
	@mkdir -p $(BIN_DIR)

$(OMP_TARGET): $(OMP_SRC) | $(BIN_DIR)
	$(CXX_OMP) $(CXXFLAGS_OMP) -o $@ $^ $(LDFLAGS_OMP)

$(MPI_TARGET): $(MPI_SRC) | $(BIN_DIR)
	$(CXX_MPI) $(CXXFLAGS_MPI) -o $@ $^ $(LDFLAGS_MPI)

$(HYB_TARGET): $(HYB_SRC) | $(BIN_DIR)
	$(CXX_HYB) $(CXXFLAGS_HYB) -o $@ $^ $(LDFLAGS_HYB)

# =========================
# Execução
# =========================
# Mantém compatibilidade: 'make run' roda a versão OMP
run: run_omp

run_omp: $(OMP_TARGET)
	@echo "Executando OMP com NUM_THREADS=$(NUM_THREADS)"
	@/usr/bin/time -p env OMP_NUM_THREADS=$(NUM_THREADS) OMP_DYNAMIC=0 ./$(OMP_TARGET)

run_mpi: $(MPI_TARGET)
	@echo "Executando MPI com PROCS=$(PROCS)"
	@/usr/bin/time -p mpirun -np $(PROCS) ./$(MPI_TARGET)

# Para o híbrido, use mpirun para >=1 processo; OMP_NUM_THREADS controla as threads
run_hybrid: $(HYB_TARGET)
	@echo "Executando HÍBRIDO com PROCS=$(PROCS) e OMP_NUM_THREADS=$(NUM_THREADS)"
	@/usr/bin/time -p env OMP_NUM_THREADS=$(NUM_THREADS) OMP_DYNAMIC=0 mpirun -np $(PROCS) ./$(HYB_TARGET)

# =========================
# Benchmarks
# =========================
bench: bench_omp

bench_omp: $(OMP_TARGET)
	@for t in $(THREADS_LIST); do \
	  echo "=== OMP_NUM_THREADS=$$t (REPS=$(REPS)) ==="; \
	  sum=0; \
	  for i in `seq 1 $(REPS)`; do \
	    tsec=$$( (env OMP_NUM_THREADS=$$t OMP_DYNAMIC=0 /usr/bin/time -p ./$(OMP_TARGET) >/dev/null) 2>&1 | awk '/real/ {print $$2}'); \
	    echo "  run $$i: $$tsec s"; \
	    sum=$$(awk "BEGIN {printf \"%.6f\", $$sum + $$tsec}"); \
	  done; \
	  avg=$$(awk "BEGIN {printf \"%.6f\", $$sum/$(REPS)}"); \
	  echo "  avg: $$avg s"; \
	  echo ""; \
	done

bench_mpi: $(MPI_TARGET)
	@for p in $(PROCS_LIST); do \
	  echo "=== MPI np=$$p (REPS=$(REPS)) ==="; \
	  sum=0; \
	  for i in `seq 1 $(REPS)`; do \
	    tsec=$$( (/usr/bin/time -p mpirun -np $$p ./$(MPI_TARGET) >/dev/null) 2>&1 | awk '/real/ {print $$2}'); \
	    echo "  run $$i: $$tsec s"; \
	    sum=$$(awk "BEGIN {printf \"%.6f\", $$sum + $$tsec}"); \
	  done; \
	  avg=$$(awk "BEGIN {printf \"%.6f\", $$sum/$(REPS)}"); \
	  echo "  avg: $$avg s"; \
	  echo ""; \
	done

# =========================
# Benchmark híbrido (macOS/Linux)
# - macOS: sem binding e sem PE= (afinidade não suportada)
# - Linux: usa slot:PE=$t e --bind-to core
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
  HYB_MPIRUN_FLAGS = --map-by slot --bind-to none --oversubscribe
  HYB_OMP_ENV     = OMP_PROC_BIND=FALSE
else
  HYB_MPIRUN_FLAGS = --map-by slot:PE=$$t --bind-to core
  HYB_OMP_ENV     =
endif

bench_hybrid: $(HYB_TARGET)
	@for p in $(PROCS_LIST); do \
	  for t in $(THREADS_LIST); do \
	    echo "=== HÍBRIDO np=$$p, OMP_NUM_THREADS=$$t (REPS=$(REPS)) ==="; \
	    sum=0; \
	    for i in `seq 1 $(REPS)`; do \
	      tsec=$$( (/usr/bin/time -p env OMP_NUM_THREADS=$$t OMP_DYNAMIC=0 $(HYB_OMP_ENV) \
	                   mpirun -np $$p $(HYB_MPIRUN_FLAGS) ./$(HYB_TARGET) >/dev/null) 2>&1 \
	               | awk '/real/ {print $$2}'); \
	      echo "  run $$i: $$tsec s"; \
	      sum=$$(awk "BEGIN {printf \"%.6f\", $$sum + $$tsec}"); \
	    done; \
	    avg=$$(awk "BEGIN {printf \"%.6f\", $$sum/$(REPS)}"); \
	    echo "  avg: $$avg s"; \
	    echo ""; \
	  done; \
	done

# =========================
# Limpeza
# =========================
clean:
	rm -rf $(BIN_DIR)
