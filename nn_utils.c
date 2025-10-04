#include <stdlib.h>
#include <math.h>
#include "nn_utils.h"

double sigmoid(double x) { return 1.0 / (1.0 + exp(-x)); }

double softmax(double *v, int n) {
    double sum = 0.0;
    for (int i=0; i<n; i++) sum += exp(v[i]);
    for (int i=0; i<n; i++) v[i] = exp(v[i]) / sum;
    return sum;
}

double rand_weight(void) {
    return (rand() / (double)RAND_MAX) * 2 - 1;
}
