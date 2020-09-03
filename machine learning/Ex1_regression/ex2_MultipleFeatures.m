clear ; close all; clc
%% 数据集导入
data = importdata('D:\CODE\matlab\machine learning\Ex1_regression\ex1data2.txt');
X = data(:, 1:2);
y = data(:, 3);
m = length(y);
%% ================ 特征归一化 ================
%特征归一化
[X,mu,sigma] = featureNormalize(X);

% 添加一列特征值X0=1
X = [ones(m, 1) X];


%% ================ 梯度下降 ================
% Choose some alpha value
alpha = 0.01;
num_iters =1;

% 初始化theta并运行梯度下降
theta = zeros(3, 1);
[theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters);
fprintf('进行梯度下降 ...\n');
% 画出迭代收敛图
figure;
plot(1:numel(J_history), J_history, '-b', 'LineWidth', 2);
xlabel('Number of iterations');
ylabel('Cost J');
% 显示梯度下降结果
fprintf('已达到局部最小值 \n');
fprintf('此时\nθ0= %f ,\nθ1= %f,\nθ2= %f \n', theta(1),theta(2),theta(3));
fprintf('\n');
% 
% %% ================ 预测价格 ================
% area=input('请输入面积:');
% room=input('请输入房间个数:');
% 
% price = [1 (([area room]-mu) ./ sigma)] * theta ;
% 
% fprintf('预测价格为 %f\n' ,price);
% fprintf('\n');
