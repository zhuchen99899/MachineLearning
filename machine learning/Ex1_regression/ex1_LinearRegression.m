
% �������ݼ�
clear ; close all; clc
data=importdata('D:\CODE\matlab\machine learning\Ex1_regression\ex1data1.txt');
x=data(:,1);
y=data(:,2);

m=length(y);
X = [ones(m, 1), data(:,1)]; % ��x�����һ��,�γ����ݼ������Ա����h��(x)����
theta = zeros(2, 1); % ��ʼ���Ⱦ��󣨼���0=0����1=0�����Ա����h��(x)����

% ======================= �۸�/��� ���ݼ�ͼ�� =======================
plot(x, y, 'rx', 'MarkerSize', 10); 
ylabel('�۸�(Profit in $10,000s)');       %�趨y�����
xlabel('���ݴ�С(Population of City in 10,000s)');   %�趨x�����


%% ======================= ���ۺ������ݶ��½��Լ�Ԥ������ =======================
%�ݶ��½�����
iterations = 1500; %��������
alpha = 0.01; %ѧϰ��

% ���ָ��(��0,��1)����ʱ�Ĵ��ۺ���
% J = computeCost(X, y, theta);% ��ʼ�����ۺ���������0=0����1=0��
% fprintf('��0=0����1=0ʱ�����ۺ���J(��) =%f\n', J);
% J = computeCost(X, y, [-1 ; 2]);% ��0=-1����1=2ʱ�Ĵ��ۺ���
% fprintf('��0=-1����1=2ʱ�����ۺ���J(��) =%f\n', J);

%�����ݶ��½��㷨
theta = gradientDescent(X, y, theta, alpha, iterations);% �����ݶ��½����Ӧ�0=0,��1=0��ʼ�����ݶ��½�
fprintf('�ݶ��½��ѵ���ֲ����Ž⣡');
fprintf('��ʱ��0=%f , ��1=%f\n', theta(1),theta(2));

%�������Իع��������(���Ȼ�ѵ����ͼ) ����Ϊ���ݼ�x��������Ϊh��(x)
hold on;
plot(x, X*theta, '-') 
legend('ѵ����', '���Իع�')
hold off;



% %����Ԥ�� 
% predict1 = [1, 3.5] *theta;
% fprintf('���ݴ�С = 35,000, Ԥ�Ƽ۸�Ϊ %f\n',...
%     predict1*10000);
% predict2 = [1, 7] * theta;
% fprintf('���ݴ�С = 70,000, Ԥ�Ƽ۸�Ϊ %f\n',...
%     predict2*10000);

% Ӧ��
temp0=input('�����뷿�����\n');
temp=temp0/1000;
predict=[1,temp]*theta;
fprintf('���ݴ�С = %f, Ԥ�Ƽ۸�Ϊ %f\n',temp0,predict*10000)


%% ======================= ����ͼ�� =======================
% X��Y�ḳֵ
% theta0_vals = linspace(-10, 10, 100);
% theta1_vals = linspace(-1, 4, 100);

% Z�ḳֵ
% J_vals = zeros(length(theta0_vals), length(theta1_vals));
% for i = 1:length(theta0_vals)
%      for j = 1:length(theta1_vals) 
%         t = [theta0_vals(i); theta1_vals(j)]; 
%         J_vals(i,j) = computeCost(X, y, t); 
%     end
% end
% J_vals = J_vals';

% %���ʹ��ۺ���ͼ�λ���
% figure;
% surf(theta0_vals, theta1_vals, J_vals)
% xlabel('\theta_0'); ylabel('\theta_1');

% % �ȸ���ͼ����
% figure;
% contour(theta0_vals, theta1_vals, J_vals, logspace(-2, 3, 20))
% xlabel('\theta_0'); ylabel('\theta_1');



