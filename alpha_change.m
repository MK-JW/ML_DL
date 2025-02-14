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
rho = 0.01; %选择小一点的rho，通过数据得到
sigma = 1;  %选择大一点的sigma，通过数据得到
[m,n] = size(x_train);
% alpha = 0.5; %根据数据手动调整,<
tolerance = 1e-6;
loss = [];
k = 0;
w_k = ones(1,n)*0.1; b_k = ones(m,1)*0.1;
y_prediction = x_train*w_k' + b_k;
cost = (1/(2*m))*sum((y_prediction - y_train).^2);
dw_k = -(1/m)*(y_prediction - y_train)'*x_train;
db_k = ones(m,1)*(-(1/m)*(y_prediction - y_train)'*ones(m,1));
x_current = [w_k,b_k'];
d_current = [dw_k,db_k'];
alpha = Armijo_wolfe(y_train, x_train, x_current, d_current, rho, sigma);
loss = [loss; cost];
k = 1;
w_previous = w_k;
b_previous = b_k;
w_k = w_previous + alpha*dw_k;
b_k = b_previous + alpha*db_k;
y_prediction = x_train*w_k' + b_k;
cost = (1/(2*m))*sum((y_prediction - y_train).^2);
dw_k = -(1/m)*(y_prediction - y_train)'*x_train;
db_k = ones(m,1)*(-(1/m)*(y_prediction - y_train)'*ones(m,1));
x_current = [w_k,b_k'];
d_current = [dw_k,db_k'];
alpha = Armijo_wolfe(y_train, x_train, x_current, d_current, rho, sigma);
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
    db_k = ones(m,1)*(-(1/m)*(y_prediction - y_train)'*ones(m,1));
    x_current = [w_k,b_k'];
    d_current = [dw_k,db_k'];
    alpha = Armijo_wolfe(y_train, x_train, x_current, d_current, rho, sigma);
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
%% function Armijo_wolfe
function [alpha_acceptable] = Armijo_wolfe(y_train, x_train, x_current, d_current, rho, sigma)
    k_max = 1000;
    [m,n] = size(x_train);
    k = 0;
    alpha_lower_k = 0;
    alpha_upper_k = 1;
    x_alpha_lower_k = x_current + alpha_lower_k*d_current;
    y_prediction = x_train*x_alpha_lower_k(1,1:n)'+ x_alpha_lower_k(1,n+1:end)';
    gw_k = (1/m)*(y_prediction - y_train)'*x_train;
    gb_k = ones(m,1)*((1/m)*(y_prediction - y_train)'*ones(m,1));
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
        y_prediction = x_train*x_alpha_k(1,1:n)'+ x_alpha_k(1,n+1:end)';
        gw_k = (1/m)*(y_prediction - y_train)'*x_train;
        gb_k = ones(m,1)*((1/m)*(y_prediction - y_train)'*ones(m,1));
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
        alpha_acceptable = NaN;
    end
end