clc
clear
%% data process
data_f = readmatrix('world-happiness-report-2017.csv');
data = data_f(:,3:end);
y_train = data(:,1);
x_train = data(:,4);
figure(1)
scatter(x_train,y_train,10,'r','filled');
%% linear regression
[m,n] = size(x_train);
alpha = 0.8; %根据数据手动调整
tolerance = 1*10^-6;
loss = [];
k = 0;
w_k = 0; b_k = 0;
y_prediction = w_k*x_train + b_k;
cost = (1/(2*m))*sum((y_prediction - y_train).^2);
dw_k = -(1/m)*(y_prediction - y_train)'*x_train;
db_k = -(1/m)*(y_prediction - y_train)'*ones(m,n);
loss = [loss; cost];
k = 1;
w_previous = w_k;
b_previous = b_k;
w_k = w_previous + alpha*dw_k;
b_k = b_previous + alpha*db_k;
y_prediction = w_k*x_train + b_k;
cost = (1/(2*m))*sum((y_prediction - y_train).^2);
dw_k = -(1/m)*(y_prediction - y_train)'*x_train;
db_k = -(1/m)*(y_prediction - y_train)'*ones(m,n);
loss = [loss; cost];
while(norm(w_k - w_previous)>tolerance || norm(b_k - b_previous)>tolerance)
    k = k+1;
    w_previous = w_k;
    b_previous = b_k;
    w_k = w_previous + alpha*dw_k';
    b_k = b_previous + alpha*db_k';
    y_prediction = w_k*x_train + b_k;
    cost = (1/(2*m))*sum((y_prediction - y_train).^2);
    dw_k = -(1/m)*(y_prediction - y_train)'*x_train;
    db_k = -(1/m)*(y_prediction - y_train)'*ones(m,n);
    loss = [loss; cost];
end
figure(1)
hold on
x = 0:0.01:2;
y = w_k*x + b_k;
plot(x,y,'k','linewidth',1.5)
hold off
figure(2)
q = max(size(loss));
x_loss = 1:q; y_loss = loss;
plot(x_loss,y_loss,'b','linewidth',1.5)

gti 