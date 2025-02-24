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
rho = 0.01;
sigma = 1;
[m,n] = size(x_train);
% alpha = 0.8; %根据数据手动调整
tolerance = 1e-6;
w = [];
b = [];
loss = [];
k = 0;
w_k = 0; b_k = 0;   
y_prediction = w_k*x_train + b_k;
cost = (1/(2*m))*sum((y_prediction - y_train).^2);
dw_k = -(1/m)*(y_prediction - y_train)'*x_train;
db_k = -(1/m)*(y_prediction - y_train)'*ones(m,n);
w = [w; w_k];
b = [b; b_k];
x_current = [w_k,b_k];
d_current = [dw_k,db_k];
alpha = Armijo_wolfe(y_train, x_train, x_current, d_current, rho, sigma);
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
w = [w; w_k];
b = [b; b_k];
x_current = [w_k,b_k];
d_current = [dw_k,db_k];
alpha = Armijo_wolfe(y_train, x_train, x_current, d_current, rho, sigma);
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
    w = [w; w_k];
    b = [b; b_k];
    x_current = [w_k,b_k];
    d_current = [dw_k,db_k];
    alpha = Armijo_wolfe(y_train, x_train, x_current, d_current, rho, sigma);
    loss = [loss; cost];
end
figure(1)
hold on
% x = 0:0.01:2;
x = [min(x_train):0.1:max(x_train)];
y = w_k*x + b_k;
plot(x,y,'k','linewidth',1.5)
hold off
figure(2)
q = max(size(loss));
x_loss = 1:q; y_loss = loss;
plot(x_loss,y_loss,'b','linewidth',1.5)
xlabel("interation")
ylabel("loss value")
figure(3)
[X, Y] = meshgrid(0:0.3:30);
[r,s] = size(X);
t = 1;
for i = 1:r
    loss_p = [];
    for j = 1:s
        y_p = X(1,i)*x_train + Y(j,i);
        cost = (1/(2*m))*sum((y_p - y_train).^2);
        loss_p = [loss_p, cost];
    end
    loss_t(t,:) = loss_p;
    t = t+1;
end
surf(X,Y,loss_t,'FaceAlpha',0.5)
hold on
plot3(w,b,loss ,'b-o','linewidth',2)
hold off
% 跑个示例代码学一下surf
% [X,Y] = meshgrid(-5:.5:5);
% Z = Y.*sin(X) - X.*cos(Y);
% s = surf(X,Y,Z,'FaceAlpha',0.5);
%% Function Armijo wolfe
function [alpha_acceptable] = Armijo_wolfe(y_train, x_train, x_current, d_current, rho, sigma)
    k_max = 1000;
    [m,n] = size(x_train);
    k = 0;
    alpha_lower_k = 0;
    alpha_upper_k = 1;
    x_alpha_lower_k = x_current + alpha_lower_k*d_current;
    y_prediction = x_alpha_lower_k(1)*x_train + x_alpha_lower_k(2);
    gw_k = (1/m)*(y_prediction - y_train)'*x_train;
    gb_k = (1/m)*(y_prediction - y_train)'*ones(m,n);
    f_x_alpha_lower_k = (1/(2*m))*sum((y_prediction - y_train).^2);
    df_x_alpha_lower_k = d_current*([gw_k, gb_k'])';
    f_x_alpha_lower_0 = f_x_alpha_lower_k;
    df_x_alpha_lower_0 = df_x_alpha_lower_k;
    tolerance = 10^-15;
%     if(df_x_alpha_lower_0> 0)
%         df_x_alpha_lower_0 = -df_x_alpha_lower_0;
%     end
    if(abs(df_x_alpha_lower_k) >tolerance)
        alpha_k = -2*f_x_alpha_lower_k/df_x_alpha_lower_k;
%         alpha_k = 1;
    else
        alpha_k = 1;
    end
    for k = 1:k_max
        x_alpha_k = x_current + alpha_k*d_current;
        y_prediction = x_alpha_k(1)*x_train+ x_alpha_k(2);
        gw_k = (1/m)*(y_prediction - y_train)'*x_train;
        gb_k = (1/m)*(y_prediction - y_train)'*ones(m,n);
        f_x_alpha_k = (1/(2*m))*sum((y_prediction - y_train).^2);
        df_x_alpha_k = d_current*([gw_k, gb_k'])';
        Armijo_condition = f_x_alpha_k - f_x_alpha_lower_0 - rho*df_x_alpha_lower_0*alpha_k;
        wolfe_condition =  abs(df_x_alpha_k) - sigma*abs(df_x_alpha_lower_0);
        if(Armijo_condition <=0)
            if(wolfe_condition <=0)
                alpha_acceptable = alpha_k;
                break;
            else
                if(df_x_alpha_k <0)
                    delta_alpha_k = (alpha_k - alpha_lower_k)*df_x_alpha_k/(df_x_alpha_lower_k - df_x_alpha_k);
                    if(delta_alpha_k <=0)
                        alpha_k_temp = alpha_k - delta_alpha_k;
                    else
                        alpha_k_temp = alpha_k + delta_alpha_k;
                    end
                    alpha_lower_k = alpha_k;
                    f_x_alpha_lower_k = f_x_alpha_k;
                    df_x_alpha_lower_k = df_x_alpha_k;
                    alpha_k = alpha_k_temp;
                else
                    if(alpha_k<alpha_upper_k)
                        alpha_upper_k = alpha_k;
                    end
                    alpha_k_temp = alpha_lower_k - (1/2)*((alpha_k - alpha_lower_k)^2*df_x_alpha_lower_k)/(f_x_alpha_k - ...
                    f_x_alpha_lower_k - df_x_alpha_lower_k*(alpha_k - alpha_lower_k));
                    alpha_k = alpha_k_temp;
                end
            end
        else
            if(alpha_k <alpha_upper_k)
                alpha_upper_k = alpha_k;
            end
            alpha_k_temp = alpha_lower_k - (1/2)*((alpha_k - alpha_lower_k)^2*df_x_alpha_lower_k)/(f_x_alpha_k - ...
                f_x_alpha_lower_k - df_x_alpha_lower_k*(alpha_k - alpha_lower_k));
            alpha_k = alpha_k_temp;
        end
        if(alpha_upper_k - alpha_lower_k <tolerance)
            alpha_acceptable = alpha_k;
            break;
        end
    end
    if((Armijo_condition >0)||(wolfe_condition>0))
%         alpha_acceptable = 0.01;
        alpha_acceptable = NaN;
    end
end