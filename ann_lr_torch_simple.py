import numpy as np
import torch
import matplotlib.pyplot as plt

torch.manual_seed(1)

# 데이터
n = 17
x_input = np.array([3.3,4.4,5.5,6.71,6.93,4.168,9.779,6.182,7.59,2.167,7.042,10.791,5.313,7.997,5.654,9.27,3.1]).reshape(n,1)
y_input = np.array([1.7,2.76,2.09,3.19,1.694,1.573,3.366,2.596,2.53,1.221,2.827,3.465,1.65,2.904,2.42,2.94,1.3]).reshape(n,1)

x_train = torch.FloatTensor(x_input)
y_train = torch.FloatTensor(y_input)

learning_rate = 0.01
training_epochs = 5000
display_step = 50

# 모델 초기화
W = torch.zeros(1, requires_grad=True)  
b = torch.zeros(1, requires_grad=True)  
    # size = 1, requires_grad=True If autograd record operations on returned tensor
    
# optimizer 설정
optimizer = torch.optim.SGD([W, b], lr=learning_rate)

for epoch in range(training_epochs):
    
    hypothesis = x_train * W + b
    
    cost = torch.mean((hypothesis - y_train) ** 2)
    
    optimizer.zero_grad()
    cost.backward()             # 미분 
    optimizer.step()            # 계수 업데이트 - W, b
    
    if epoch % display_step == 0:
        print('Epoch {:4d}/{} W: {:.3f}, b: {:.3f} Cost: {:.6f}'.format(
            epoch, training_epochs, W.item(), b.item(), cost.item()
        ))

x = x_train.numpy().reshape(n)
y = y_train.numpy().reshape(n)
y2 = x * W.item() + b.item() 
plt.plot(x, y, 'ro', label='Original data')
plt.plot(x, y2, label='Fitted line by ANN - Torch - Simple')
plt.legend()
plt.show()