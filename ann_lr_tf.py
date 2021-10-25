#ann_lr_tf.py
import numpy as np
import matplotlib.pyplot as plt

x = np.array([3.3,4.4,5.5,6.71,6.93,4.168,9.779,6.182,7.59,2.167,7.042,10.791,5.313,7.997,5.654,9.27,3.1])
y = np.array([1.7,2.76,2.09,3.19,1.694,1.573,3.366,2.596,2.53,1.221,2.827,3.465,1.65,2.904,2.42,2.94,1.3])

n = len(x)

#Linear Regression by LS
W = 1/(x.sum()*x.sum() - n*(x*x).sum()) * (x.sum()*y.sum() - n*(x*y).sum())
b = 1/(x.sum()*x.sum() - n*(x*x).sum()) * (-(x*x).sum()*y.sum() + x.sum()*(x*y).sum())
# y = b0 + b1*x -> b + W*x
# ndarray * ndarray = term-by-term multication

plt.plot(x, y, 'ro', label='Original data')
plt.plot(x, np.array(W * x + b), label='Fitted line by LS')
plt.legend()
plt.show()

#Linear Regression by ANN - Tensorflow
import tensorflow as tf

W = tf.Variable(np.random.randn(), name = 'W')
b = tf.Variable(np.random.randn(), name = 'b')
# 초기화: W, b를 구하는 것이 목적
# 
learning_rate = 0.01
training_epochs = 5000
display_step = 50

# Define Linear Regression Model
def linear_regression(x):
    return W*x + b
    # y_pred = b + W*x
    # | y_true - y_pred |^2 to be minimized
    
# Define cost Function
def Cost(y_pred, y_true):
    return tf.reduce_sum(tf.pow(y_pred-y_true, 2)) / (2 * n)
    # |y_pred - y_true |^2

# Gradient Descent Optimizer
optimizer = tf.optimizers.SGD(learning_rate)

#define optimization of learning
def run_optimization():
    with tf.GradientTape() as g:
        pred = linear_regression(x)
        loss = Cost(pred, y)
        
    gradients = g.gradient(loss, [W, b])
    optimizer.apply_gradients(zip(gradients, [W,b]))
    
for step in range(training_epochs):
    run_optimization()
    
    if step % display_step == 0:
        pred = linear_regression(x)
        loss = Cost(pred, y)
        print("step: %i, loss: %f, W: %f, b: %f" % (step, loss, W.numpy(), b.numpy()))
        
plt.plot(x, y, 'ro', label='Original data')
plt.plot(x, np.array(W * x + b), label='Fitted line by ANN - TensorFlow')
plt.legend()
plt.show()


