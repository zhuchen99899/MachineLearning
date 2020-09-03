%% ���ݼ�����
data = importdata('D:\CODE\matlab\�����¶�\trainingdata.txt');
X = data(:, 1:3);
y = data(:, 4);
m = length(y);

% ���һ������ֵX0=1 ,�γ�ѵ����
X = [ones(m, 1) X];

%% ================ �ݶ��½� ================
% ѡ��ѧϰ�ʣ��ݶ��½���������
alpha = 0.0000001;
num_iters =100000;

% ��ʼ��theta�������ݶ��½�
theta = zeros(4, 1);
[theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters);
fprintf('�����ݶ��½� ...\n');
% ������������ͼ
figure;
plot(1:numel(J_history), J_history, '-b', 'LineWidth', 2);
xlabel('Number of iterations');
ylabel('Cost J');
% ��ʾ�ݶ��½����
fprintf('�Ѵﵽ�ֲ���Сֵ \n');
fprintf('��ʱ\n��0= %f ,\n��1= %f,\n��2= %f \n��3= %f \n', theta(1),theta(2),theta(3),theta(4));
fprintf('\n');
%% ================ Ԥ��۸� ================
distance=input('����:');
areatemperature=input('�����¶�:');
testtemperature=input('�����¶�:');
price = [1 distance areatemperature testtemperature] * theta ;

fprintf('Ԥ���¶�Ϊ %f\n' ,price);
fprintf('\n');