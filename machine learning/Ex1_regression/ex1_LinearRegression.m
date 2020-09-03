
% 导入数据集
clear ; close all; clc
data=importdata('D:\CODE\matlab\machine learning\Ex1_regression\ex1data1.txt');
x=data(:,1);
y=data(:,2);

m=length(y);
X = [ones(m, 1), data(:,1)]; % 在x中添加一列,形成数据集矩阵，以便后续hθ(x)计算
theta = zeros(2, 1); % 初始化θ矩阵（即θ0=0，θ1=0），以便后续hθ(x)计算

% ======================= 价格/面积 数据集图像 =======================
plot(x, y, 'rx', 'MarkerSize', 10); 
ylabel('价格(Profit in $10,000s)');       %设定y轴标题
xlabel('房屋大小(Population of City in 10,000s)');   %设定x轴标题


%% ======================= 代价函数，梯度下降以及预测数据 =======================
%梯度下降设置
iterations = 1500; %迭代次数
alpha = 0.01; %学习率

% 输出指定(θ0,θ1)参数时的代价函数
% J = computeCost(X, y, theta);% 初始化代价函数（即θ0=0，θ1=0）
% fprintf('θ0=0，θ1=0时，代价函数J(θ) =%f\n', J);
% J = computeCost(X, y, [-1 ; 2]);% θ0=-1，θ1=2时的代价函数
% fprintf('θ0=-1，θ1=2时，代价函数J(θ) =%f\n', J);

%运行梯度下降算法
theta = gradientDescent(X, y, theta, alpha, iterations);% 进行梯度下降，从θ0=0,θ1=0开始进行梯度下降
fprintf('梯度下降已到达局部最优解！');
fprintf('此时θ0=%f , θ1=%f\n', theta(1),theta(2));

%绘制线性回归拟合曲线(需先绘训练集图) 横坐为数据集x，纵坐标为hθ(x)
hold on;
plot(x, X*theta, '-') 
legend('训练集', '线性回归')
hold off;



% %数据预测 
% predict1 = [1, 3.5] *theta;
% fprintf('房屋大小 = 35,000, 预计价格为 %f\n',...
%     predict1*10000);
% predict2 = [1, 7] * theta;
% fprintf('房屋大小 = 70,000, 预计价格为 %f\n',...
%     predict2*10000);

% 应用
temp0=input('请输入房屋面积\n');
temp=temp0/1000;
predict=[1,temp]*theta;
fprintf('房屋大小 = %f, 预计价格为 %f\n',temp0,predict*10000)


%% ======================= 绘制图像 =======================
% X，Y轴赋值
% theta0_vals = linspace(-10, 10, 100);
% theta1_vals = linspace(-1, 4, 100);

% Z轴赋值
% J_vals = zeros(length(theta0_vals), length(theta1_vals));
% for i = 1:length(theta0_vals)
%      for j = 1:length(theta1_vals) 
%         t = [theta0_vals(i); theta1_vals(j)]; 
%         J_vals(i,j) = computeCost(X, y, t); 
%     end
% end
% J_vals = J_vals';

% %碗型代价函数图形绘制
% figure;
% surf(theta0_vals, theta1_vals, J_vals)
% xlabel('\theta_0'); ylabel('\theta_1');

% % 等高线图绘制
% figure;
% contour(theta0_vals, theta1_vals, J_vals, logspace(-2, 3, 20))
% xlabel('\theta_0'); ylabel('\theta_1');



