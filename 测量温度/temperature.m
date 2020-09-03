%% 数据集导入
data = importdata('D:\CODE\matlab\测量温度\trainingdata.txt');
X = data(:, 1:3);
y = data(:, 4);
m = length(y);

% 添加一列特征值X0=1 ,形成训练集
X = [ones(m, 1) X];

%% ================ 梯度下降 ================
% 选择学习率，梯度下降迭代次数
alpha = 0.0000001;
num_iters =100000;

% 初始化theta并运行梯度下降
theta = zeros(4, 1);
[theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters);
fprintf('进行梯度下降 ...\n');
% 画出迭代收敛图
figure;
plot(1:numel(J_history), J_history, '-b', 'LineWidth', 2);
xlabel('Number of iterations');
ylabel('Cost J');
% 显示梯度下降结果
fprintf('已达到局部最小值 \n');
fprintf('此时\nθ0= %f ,\nθ1= %f,\nθ2= %f \nθ3= %f \n', theta(1),theta(2),theta(3),theta(4));
fprintf('\n');
%% ================ 预测价格 ================
distance=input('距离:');
areatemperature=input('环境温度:');
testtemperature=input('测量温度:');
price = [1 distance areatemperature testtemperature] * theta ;

fprintf('预测温度为 %f\n' ,price);
fprintf('\n');