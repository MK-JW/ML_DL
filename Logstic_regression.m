clc
clear
%% data process
data_t = readcell('bankloan.csv', 'VariableNamingRule', 'preserve');
[m,n] = size(data_t);
data_train = data_t(1:700,:);
data_test = data_t(701:m,:);
y_train = data_train(2:700,end);
x_train = cell2mat(data_train(2:700,4:9));
x_test = cell2mat(data_test(1:end,4:9));
y_train = cellfun(@(x) strcmp(x,'是'),y_train);
y_one = sum(y_train);
is_ave = y_one/max(size(y_train));
disp(is_ave);
%% gradient descend
[q,p] = size(x_train);
loss = [];
alpha = 0.001;
tolerance = 1e-6;
k = 0;
w_k = zeros(p,1);
b_k = 0;
z = x_train*w_k + b_k;
y_prediction = sigmoid(z);
cost = -(1/q)*(log(y_prediction)'*y_train + log(1-y_prediction)'*(1-y_train));
dw_k = -(1/q)*(y_prediction - y_train)'*x_train;
db_k = -(1/q)*(y_prediction - y_train)'*ones(q,1);                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         
loss = [loss, cost];
k = 1;
w_previous = w_k;
b_previous = b_k;
w_k = w_previous + alpha*dw_k';
b_k = b_previous + alpha*db_k;
z = x_train*w_k + b_k;
y_prediction = sigmoid(z);
cost = -(1/q)*(log(y_prediction)'*y_train + log(1-y_prediction)'*(1-y_train));
dw_k = -(1/q)*(y_prediction - y_train)'*x_train;
db_k = -(1/q)*(y_prediction - y_train)'*ones(q,1);
loss = [loss; cost];
while(norm(w_k - w_previous)>tolerance || norm(b_k - b_previous)>tolerance)
    k = k+1;
    w_previous = w_k;
    b_previous = b_k;
    w_k = w_previous + alpha*dw_k';
    b_k = b_previous + alpha*db_k;
    z = x_train*w_k + b_k;
    y_prediction = sigmoid(z);
    cost = -(1/m)*(log(y_prediction)'*y_train + log(1-y_prediction)'*(1-y_train));
    dw_k = -(1/m)*(y_prediction - y_train)'*x_train;
    db_k = -(1/m)*(y_prediction - y_train)'*ones(q,1);
    loss = [loss; cost];
end
y_prediction = y_prediction >0.5;
result = [y_train, y_prediction];
if(is_ave>0.45 && is_ave<0.55)
    accuracy = sum(y_prediction == y_train)/q;
    disp(accuracy);
else
    tp = sum((y_prediction == y_train) & (y_prediction == 1));
    fp = sum((y_prediction ~= y_train) & (y_prediction == 1));
    fn = sum((y_prediction ~= y_train) & (y_prediction == 0));
    tn = sum((y_prediction == y_train) & (y_prediction == 0));
    precision = tp/(tp + fp); %精确率
    recall = tp/(tp + fn); %召回率
    f1 = 2*(precision*recall)/(precision + recall); %F1分数，平衡精确率与召回率
    disp(precision)
    disp(recall)
    disp(f1)
end
%% visualization
% s = max(size(loss));
s = 0:10000;
figure(1)
plot(s,loss(1:max(size(s)),:),'b','linewidth',1.5)
%% sigmoid
function sigmoid = sigmoid(x)

sigmoid = 1./(1+exp(-x));

end