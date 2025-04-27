import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.linear = nn.Linear(10, 1)

    def forward(self, x):
        return self.linear(x)


def gradient_inversion_attack(model, device, num_pixels, learning_rate, num_iterations):
    """
    Gradient Inversion Attack을 수행합니다.

    Args:
        model (nn.Module): 공격 대상 모델
        device (torch.device): GPU 또는 CPU 장치
        num_pixels (int): 입력 이미지의 픽셀 수 (예: 10)
        learning_rate (float): 학습률
        num_iterations (int): 반복 횟수

    Returns:
        torch.Tensor: 재구성된 입력 데이터
    """

    # 입력 데이터 초기화
    input_data = torch.randn(1, num_pixels, device=device, requires_grad=True)

    # 옵티마이저 설정
    optimizer = optim.Adam([input_data], lr=learning_rate)

    # 공격 수행
    model.eval()  # 모델을 평가 모드로 설정
    for i in range(num_iterations):
        # 입력 데이터를 모델에 통과
        output = model(input_data)

        # 손실 계산 (예: MSE 손실)
        loss = nn.MSELoss()(output, torch.zeros_like(output))  # 목표는 0에 가까워지도록

        # 손실 역전파
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print(f"Iteration {i+1}/{num_iterations}, Loss: {loss.item()}")

    return input_data.detach()


if __name__ == '__main__':
    # 모델, 장치 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleModel().to(device)

    # 공격 파라미터 설정
    num_pixels = 10
    learning_rate = 0.1
    num_iterations = 1000

    # Gradient Inversion Attack 수행
    reconstructed_input = gradient_inversion_attack(model, device, num_pixels, learning_rate, num_iterations)

    print("Reconstructed Input:", reconstructed_input)
