//
// Created by potados on 2025-02-13.
//

#include "random.h"
#include <stdlib.h>

// 랜덤 초기화 함수 (-1.0 ~ 1.0 범위)
double random_weight() {
    return ((double)rand() / RAND_MAX) * 2.0 - 1.0;
}