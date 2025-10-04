#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "run_model.h"
#include "nn_utils.h"
#include "tokenizer.h"

#define INPUTS 27
#define OUTPUTS 27

static void one_hot(int idx, double *v, int size) {
    for (int i = 0; i < size; i++) v[i] = 0.0;
    v[idx] = 1.0;
}

void run_model(const char *weights_path, const char *prompt) {
    FILE *wf = fopen(weights_path, "rb");
    if (!wf) { perror("weights"); exit(1); }
    double W[INPUTS][OUTPUTS];
    fread(W, sizeof(W), 1, wf);
    fclose(wf);

    int token = char_to_token(prompt[0]);
    printf("%c", prompt[0]);

    for (int step=0; step<50; step++) {
        double x[INPUTS], y_pred[OUTPUTS];
        one_hot(token, x, INPUTS);

        for (int j=0;j<OUTPUTS;j++) {
            double sum = 0.0;
            for (int i=0;i<INPUTS;i++)
                sum += x[i]*W[i][j];
            y_pred[j] = sum;
        }
        softmax(y_pred, OUTPUTS);

        // pick max-prob token
        int next = 0;
        double best = y_pred[0];
        for (int j=1;j<OUTPUTS;j++)
            if (y_pred[j] > best) { best = y_pred[j]; next = j; }

        char c = token_to_char(next);
        printf("%c", c);
        token = next;
    }
    printf("\n");
}
