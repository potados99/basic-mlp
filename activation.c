//
// Created by potados on 2025-02-13.
//

#include "activation.h"
#include <math.h>

// 시그모이드 함수
double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

// 시그모이드 미분
double sigmoid_derivative(double x) {
    return x * (1.0 - x);
}
