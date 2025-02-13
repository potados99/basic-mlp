#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "random.h"
#include "activation.h"

#define HIDDEN_SIZE 4
#define OUTPUT_SIZE 1

#define LEARNING_RATE 0.009f
#define EPOCHS 1000000

#define INPUT_SIZE 2
#define DATASET_SIZE 5

// 학습 데이터 (기온, 습도 -> 비 올 확률, 정규화됨)
double X[DATASET_SIZE][INPUT_SIZE] = {
    {0.0008155970599585504, 0.3671065876906898},
    {0.09438936020373064, 0.893161212420147},
    {0.035482817128173895, 0.4482304623480121},
    {0.38440866478481384, 0.5153324993799973},
    {0.30, 0.70}
};
double y[DATASET_SIZE] = {1, 0, 0, 0, 1};

// 가중치 및 편향
double W1[INPUT_SIZE][HIDDEN_SIZE];
double b1[HIDDEN_SIZE];
double W2[HIDDEN_SIZE][OUTPUT_SIZE];
double b2[OUTPUT_SIZE];

// 예측 함수 (Forward Propagation만 수행)
void predict(double input[INPUT_SIZE], double output[OUTPUT_SIZE]) {
    double z1[HIDDEN_SIZE], a1[HIDDEN_SIZE];

    for (int j = 0; j < HIDDEN_SIZE; j++) {
        z1[j] = b1[j];
        for (int k = 0; k < INPUT_SIZE; k++)
            z1[j] += input[k] * W1[k][j];
        a1[j] = sigmoid(z1[j]);
    }

    for (int j = 0; j < OUTPUT_SIZE; j++) {
        output[j] = b2[j];
        for (int k = 0; k < HIDDEN_SIZE; k++)
            output[j] += a1[k] * W2[k][j];
        output[j] = sigmoid(output[j]);
    }
}

// 학습 함수 (Training)
void train() {
    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        double total_loss = 0.0;

        for (int i = 0; i < DATASET_SIZE; i++) {
            // --- 순전파 ---
            double z1[HIDDEN_SIZE], a1[HIDDEN_SIZE];
            for (int j = 0; j < HIDDEN_SIZE; j++) {
                z1[j] = b1[j];
                for (int k = 0; k < INPUT_SIZE; k++)
                    z1[j] += X[i][k] * W1[k][j];
                a1[j] = sigmoid(z1[j]);
            }

            double z2[OUTPUT_SIZE], a2[OUTPUT_SIZE];
            for (int j = 0; j < OUTPUT_SIZE; j++) {
                z2[j] = b2[j];
                for (int k = 0; k < HIDDEN_SIZE; k++)
                    z2[j] += a1[k] * W2[k][j];
                a2[j] = sigmoid(z2[j]);
            }

            // --- 손실 계산 ---
            for (int o = 0; o < OUTPUT_SIZE; o++) {
                double error = y[i] - a2[o];
                total_loss += error * error;

                // --- 역전파 ---
                double d_output = error * sigmoid_derivative(a2[o]);

                double d_hidden[HIDDEN_SIZE];
                for (int j = 0; j < HIDDEN_SIZE; j++)
                    d_hidden[j] = d_output * W2[j][0] * sigmoid_derivative(a1[j]);

                // --- 가중치 업데이트 ---
                for (int o = 0; o < OUTPUT_SIZE; o++) {
                    for (int j = 0; j < HIDDEN_SIZE; j++)
                        W2[j][o] += LEARNING_RATE * d_output * a1[j];
                    b2[o] += LEARNING_RATE * d_output;
                }

                for (int j = 0; j < HIDDEN_SIZE; j++) {
                    for (int k = 0; k < INPUT_SIZE; k++)
                        W1[k][j] += LEARNING_RATE * d_hidden[j] * X[i][k];
                    b1[j] += LEARNING_RATE * d_hidden[j];
                }
            }
        }

        if (epoch % 1000 == 0) {
            printf("Epoch %d, Loss: %.4f\n", epoch, total_loss / DATASET_SIZE);
        }
    }
}

int main() {
    srand(time(0));

    // 가중치 및 편향 초기화
    for (int i = 0; i < INPUT_SIZE; i++)
        for (int j = 0; j < HIDDEN_SIZE; j++)
            W1[i][j] = random_weight();

    for (int i = 0; i < HIDDEN_SIZE; i++)
        b1[i] = random_weight();

    for (int i = 0; i < HIDDEN_SIZE; i++)
        for (int j = 0; j < OUTPUT_SIZE; j++)
            W2[i][j] = random_weight();

    for (int i = 0; i < OUTPUT_SIZE; i++)
        b2[i] = random_weight();

    // 학습 시작
    train();

    // 최종 예측 결과
    printf("\nFinal prediction results:\n");
    for (int i = 0; i < DATASET_SIZE; i++) {
        double prediction[OUTPUT_SIZE];
        predict(X[i], prediction);

        printf("Input: [%.4f, %.4f] -> Prediction: ", X[i][0], X[i][1]);
        for (int j = 0; j < OUTPUT_SIZE; j++) {
            printf("%.4f ", prediction[j]);
        }
        printf("\n");
    }

    return 0;
}
