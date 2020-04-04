function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

% Cost calculation
hOfTheta = X * theta;
square_error = sum((hOfTheta - y) .^ 2);

% Don't want to regularize theta_0
temp_theta = theta;
temp_theta(1) = 0;
regularization_term = lambda * sum(temp_theta .^ 2);

J = (square_error + regularization_term) / (2 * m);

% Gradient calculation
grad = (((hOfTheta - y)' * X)' + lambda * temp_theta) / m;





% =========================================================================

grad = grad(:);

end
