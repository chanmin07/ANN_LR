% ex_lr.m
x = [3.3,4.4,5.5,6.71,6.93,4.168,9.779,6.182,7.59,2.167,7.042,10.791,5.313,7.997,5.654,9.27,3.1];
y = [1.7,2.76,2.09,3.19,1.694,1.573,3.366,2.596,2.53,1.221,2.827,3.465,1.65,2.904,2.42,2.94,1.3];
n = length(x);
b1 = 1/(sum(x)*sum(x) - n*sum(x.*x)) * (sum(x)*sum(y) - n*sum(x.*y))
b0 = 1/(sum(x)*sum(x) - n*sum(x.*x)) * (-sum(x.*x)*sum(y) + sum(x)*sum(x.*y))
plot(x,y,'*')

hold on
plot(x,b1*x + b0, 'r')