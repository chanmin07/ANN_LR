import numpy as np
import matplotlib.pyplot as plt
import torch

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

#Model setup
class LinearRegressionModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)
    
our_model = LinearRegressionModel()

optimizer = torch.optim.SGD(our_model.parameters(), lr=learning_rate)
    
for epoch in range(training_epochs):
    
    #Forward Propagation
    prediction = our_model(x_train)
    
    #Compute and print loss
    cost = torch.nn.functional.mse_loss(prediction, y_train)
    
    #Backpropagation + update weights
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()
    
    if epoch % display_step == 0:
        print('epoch {}, loss {}'.format(epoch, cost.item()))
        
x = x_train.numpy().reshape(n)
y = y_train.numpy().reshape(n)
y2 = prediction.detach().numpy().reshape(n) 
plt.plot(x, y, 'ro', label='Original data')
plt.plot(x, y2, label='Fitted line by ANN - Torch - Class')
plt.legend()
plt.show()
    

