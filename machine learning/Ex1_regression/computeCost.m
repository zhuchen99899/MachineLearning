
%代价函数计算
function J = computeCost(X, y, theta)
m = length(y); 
J = 0;

J = sum((X * theta - y).^2) / (2*m);     % X(79,2)  theta(2,1)
end
