#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "train_model.h"
#include "run_model.h"

int main(int argc, char *argv[]) {
    if (argc < 2) {
        printf("Usage: %s --train <data.txt> <weights.bin> [epochs] | --run <weights.bin> <prompt>\n", argv[0]);
        return 1;
    }

    if (strcmp(argv[1], "--train") == 0 && (argc == 4 || argc == 5)) {
        int epochs = (argc == 5) ? atoi(argv[4]) : 5000; // default 5000 if not specified
        train_model(argv[2], argv[3], epochs);
    } else if (strcmp(argv[1], "--run") == 0 && argc == 4) {
        run_model(argv[2], argv[3]);
    } else {
        printf("Invalid arguments.\n");
    }
    return 0;
}
