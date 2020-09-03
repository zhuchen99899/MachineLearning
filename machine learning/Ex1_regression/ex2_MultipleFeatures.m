clear ; close all; clc
%% ���ݼ�����
data = importdata('D:\CODE\matlab\machine learning\Ex1_regression\ex1data2.txt');
X = data(:, 1:2);
y = data(:, 3);
m = length(y);
%% ================ ������һ�� ================
%������һ��
[X,mu,sigma] = featureNormalize(X);

% ���һ������ֵX0=1
X = [ones(m, 1) X];


%% ================ �ݶ��½� ================
% Choose some alpha value
alpha = 0.01;
num_iters =1;

% ��ʼ��theta�������ݶ��½�
theta = zeros(3, 1);
[theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters);
fprintf('�����ݶ��½� ...\n');
% ������������ͼ
figure;
plot(1:numel(J_history), J_history, '-b', 'LineWidth', 2);
xlabel('Number of iterations');
ylabel('Cost J');
% ��ʾ�ݶ��½����
fprintf('�Ѵﵽ�ֲ���Сֵ \n');
fprintf('��ʱ\n��0= %f ,\n��1= %f,\n��2= %f \n', theta(1),theta(2),theta(3));
fprintf('\n');
% 
% %% ================ Ԥ��۸� ================
% area=input('���������:');
% room=input('�����뷿�����:');
% 
% price = [1 (([area room]-mu) ./ sigma)] * theta ;
% 
% fprintf('Ԥ��۸�Ϊ %f\n' ,price);
% fprintf('\n');
