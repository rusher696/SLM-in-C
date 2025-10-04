#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "train_model.h"
#include "tokenizer.h"
#include "nn_utils.h"

#define INPUTS 27
#define OUTPUTS 27
#define EPOCHS 5000
#define LR 0.5

static void one_hot(int idx, double *v, int size) {
    for (int i = 0; i < size; i++) v[i] = 0.0;
    v[idx] = 1.0;
}

void train_model(const char *data_path, const char *weights_out) {
    FILE *f = fopen(data_path, "r");
    if (!f) { perror("data"); exit(1); }

    char text[4096];
    size_t n = fread(text, 1, sizeof(text)-1, f);
    fclose(f);
    text[n] = '\0';

    // weight matrix: INPUTS x OUTPUTS
    double W[INPUTS][OUTPUTS];
    srand(time(NULL));
    for (int i=0;i<INPUTS;i++)
        for (int j=0;j<OUTPUTS;j++)
            W[i][j] = rand_weight();

    for (int epoch=0; epoch<EPOCHS; epoch++) {
        double total_error = 0.0;
        for (size_t t=0; t+1<n; t++) {
            int in = char_to_token(text[t]);
            int out = char_to_token(text[t+1]);
            if (in >= INPUTS || out >= OUTPUTS) continue;

            double x[INPUTS], y_true[OUTPUTS], y_pred[OUTPUTS];
            one_hot(in, x, INPUTS);
            one_hot(out, y_true, OUTPUTS);

            // forward: y_pred = softmax(xW)
            for (int j=0;j<OUTPUTS;j++) {
                double sum = 0.0;
                for (int i=0;i<INPUTS;i++)
                    sum += x[i]*W[i][j];
                y_pred[j] = sum;
            }
            softmax(y_pred, OUTPUTS);

            // compute error & gradient
            for (int j=0;j<OUTPUTS;j++) {
                double err = y_true[j] - y_pred[j];
                total_error += 0.5*err*err;
                for (int i=0;i<INPUTS;i++)
                    W[i][j] += LR * err * x[i]; // gradient step
            }
        }

        if (epoch % 500 == 0)
            printf("Epoch %d  error=%.4f\n", epoch, total_error);
    }

    FILE *wf = fopen(weights_out, "wb");
    fwrite(W, sizeof(W), 1, wf);
    fclose(wf);
    printf("Training done! saved weights to %s\n", weights_out);
}
