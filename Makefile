CC=gcc
CFLAGS=-O2 -lm

SRC=main.c train_model.c run_model.c tokenizer.c nn_utils.c
OBJ=$(SRC:.c=.o)
OUT=slm

all: $(OUT)

$(OUT): $(OBJ)
	$(CC) $(OBJ) -o $(OUT) $(CFLAGS)

clean:
	rm -f $(OBJ) $(OUT)
