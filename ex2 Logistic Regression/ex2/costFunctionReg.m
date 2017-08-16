function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

[J, grad] = costFunction(theta, X, y, lambda);
% Should not be regularizing the theta(1) parameter (which corresponds to θ 0 ) in the code
theta_reg = [0; theta(2:end)];

J = J + 0.5 * lambda / m * sum(theta_reg .^2);

grad = grad + lambda / m * theta_reg;

% =============================================================

end
