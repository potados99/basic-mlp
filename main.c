#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

// 시그모이드 함수
double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

// 시그모이드 미분
double sigmoid_derivative(double x) {
    return x * (1.0 - x);
}

// 랜덤 초기화 함수 (-1.0 ~ 1.0 범위)
double random_weight() {
    return ((double)rand() / RAND_MAX) * 2.0 - 1.0;
}

int main() {
    srand(time(0));  // 랜덤 시드 초기화

    // 1. XOR 데이터 정의
    double X[4][2] = { {0, 0}, {0, 1}, {1, 0}, {1, 1} };  // 입력 (4개 샘플, 2개 특성)
    double y[4] = {0, 1, 1, 0};                          // 정답 출력

    // 2. 신경망 구조 정의
    int input_size = 2;
    int hidden_size = 4;
    int output_size = 1;
    double learning_rate = 0.1;

    // 가중치 및 편향 초기화
    double W1[2][4], b1[4];
    double W2[4], b2;

    for (int i = 0; i < input_size; i++)
        for (int j = 0; j < hidden_size; j++)
            W1[i][j] = random_weight();

    for (int i = 0; i < hidden_size; i++) {
        b1[i] = random_weight();
        W2[i] = random_weight();
    }
    b2 = random_weight();

    // 3. 학습 (Forward + Backpropagation)
    int epochs = 1000000;
    for (int epoch = 0; epoch < epochs; epoch++) {
        double total_loss = 0.0;

        for (int i = 0; i < 4; i++) {
            // --- 순전파 (Forward Propagation) ---
            // 입력 -> 은닉층
            double z1[4], a1[4];
            for (int j = 0; j < hidden_size; j++) {
                z1[j] = b1[j];
                for (int k = 0; k < input_size; k++)
                    z1[j] += X[i][k] * W1[k][j];
                a1[j] = sigmoid(z1[j]);
            }

            // 은닉층 -> 출력층
            double z2 = b2;
            for (int j = 0; j < hidden_size; j++)
                z2 += a1[j] * W2[j];
            double a2 = sigmoid(z2);  // 최종 예측 값

            // --- 손실(Loss) 계산 (MSE) ---
            double error = y[i] - a2;
            total_loss += error * error;

            // --- 역전파 (Backpropagation) ---
            // 출력층 오차
            double d_output = error * sigmoid_derivative(a2);

            // 은닉층 오차
            double d_hidden[4];
            for (int j = 0; j < hidden_size; j++)
                d_hidden[j] = d_output * W2[j] * sigmoid_derivative(a1[j]);

            // --- 가중치 및 편향 업데이트 ---
            for (int j = 0; j < hidden_size; j++) {
                W2[j] += learning_rate * d_output * a1[j];
            }
            b2 += learning_rate * d_output;

            for (int j = 0; j < hidden_size; j++) {
                for (int k = 0; k < input_size; k++) {
                    W1[k][j] += learning_rate * d_hidden[j] * X[i][k];
                }
                b1[j] += learning_rate * d_hidden[j];
            }
        }

        // --- 학습 진행 상황 출력 ---
        if (epoch % 1000 == 0) {
            printf("Epoch %d, Loss: %.4f\n", epoch, total_loss / 4.0);
        }
    }

    // 4. 최종 예측 결과
    printf("\nFinal prediction result:\n");
    for (int i = 0; i < 4; i++) {
        // 입력 -> 은닉층
        double z1[4], a1[4];
        for (int j = 0; j < hidden_size; j++) {
            z1[j] = b1[j];
            for (int k = 0; k < input_size; k++)
                z1[j] += X[i][k] * W1[k][j];
            a1[j] = sigmoid(z1[j]);
        }

        // 은닉층 -> 출력층
        double z2 = b2;
        for (int j = 0; j < hidden_size; j++)
            z2 += a1[j] * W2[j];
        double a2 = sigmoid(z2);

        printf("Input: [%d, %d], Prediction: %.2f\n", (int)X[i][0], (int)X[i][1], a2);
    }

    return 0;
}
