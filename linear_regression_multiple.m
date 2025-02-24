clc
clear
%% data process
data_f = readmatrix('world-happiness-report-2017.csv');
data = data_f(:,3:end);
y_train = data(:,1);
x_train = data(:,4:2:6);
figure(1)
scatter3(x_train(:,1), x_train(:,2), y_train,10,'r','filled')
xlabel('Economy')
ylabel('Health&life')
zlabel('Happiness')
grid on
%% linear regression
[m,n] = size(x_train);
alpha = 0.5; %根据数据手动调整,<
tolerance = 1*10^-6;
loss = [];
k = 0;
w_k = zeros(1,n); b_k = zeros(m,1);   
y_prediction = x_train*w_k' + b_k;
cost = (1/(2*m))*sum((y_prediction - y_train).^2);
dw_k = -(1/m)*(y_prediction - y_train)'*x_train;
db_k = -(1/m)*(y_prediction - y_train)'*ones(m,1);
loss = [loss; cost];
k = 1;
w_previous = w_k;
b_previous = b_k;
w_k = w_previous + alpha*dw_k;
b_k = b_previous + alpha*db_k;
y_prediction = x_train*w_k' + b_k;
cost = (1/(2*m))*sum((y_prediction - y_train).^2);
dw_k = -(1/m)*(y_prediction - y_train)'*x_train;
db_k = -(1/m)*(y_prediction - y_train)'*ones(m,1);
loss = [loss; cost];
while(norm(w_k - w_previous)>tolerance || norm(b_k - b_previous)>tolerance)
    k = k+1;
    w_previous = w_k;
    b_previous = b_k;
    w_k = w_previous + alpha*dw_k;
    b_k = b_previous + alpha*db_k;
    y_prediction = x_train*w_k' + b_k;
    cost = (1/(2*m))*sum((y_prediction - y_train).^2);
    dw_k = -(1/m)*(y_prediction - y_train)'*x_train;
    db_k = -(1/m)*(y_prediction - y_train)'*ones(m,1);
    loss = [loss; cost];
end
figure(1)
hold on
y = 0.2:0.01:1.8;
% y = 0.2:0.01:1.8;
x = 1.4964*y + 0.1597; %特征之间满足线性关系需要画出来
z = w_k(1)*x' + w_k(2)*y' + b_k(1);
plot3(x,y,z,'k','linewidth',1.5)
hold off
figure(2)
q = max(size(loss));
plot(1:q, loss, 'b','linewidth',1.5)
xlabel("interation")
ylabel("loss value")