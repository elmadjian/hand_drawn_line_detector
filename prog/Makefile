all: proj1

#Compiladores
CC=gcc
CXX=g++

FLAGS= -O3 -Wall

#Bibliotecas
GFTLIB  = -L$(GFT_DIR)/lib -lgft
GFTFLAGS  = -I$(GFT_DIR)/include

#Rules
libgft:
	$(MAKE) -C $(GFT_DIR)

proj1: proj1.cpp libgft
	$(CXX) $(FLAGS) $(GFTFLAGS) \
	proj1.cpp $(GFTLIB) -o proj1 -lm -lz

clean:
	$(RM) *~ *.o proj1


