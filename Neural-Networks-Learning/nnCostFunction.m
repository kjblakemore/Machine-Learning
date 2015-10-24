function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

X = [ones(m, 1) X];					% Prepend a column of ones to the input layer
yk = zeros(num_labels, 1);			% Logical vector for the output units

% Cycle through each training example, using feed forward propagation to compute the cost
for t = 1:m;
	% Fist use feed forward propagation to calculate the hypothesis
	% a2 = sigmoid(Theta1 * a1); prepend bias element to a2; hypothesis = sigmoid(Theta2 * a2);
	hyp = sigmoid(Theta2 * [1; sigmoid(Theta1 * X(t,:)')]);
	
	% Convert the label in y to the logical vector, yk, where a 1 in the ith element of yk corresponds 
	% to a label = i
	yk(y(t)) = 1;
	
	% Compute the cost of this example and add it to the total cost
	J = J + sum(-yk .* log(hyp) - (1 - yk) .* log(1 - hyp));
	
	yk(y(t)) = 0; 	% reinitialize the output vector
end

J = 1/m * J;	% The average, non-regularized cost

% Add regularization to the cost.
J = J + lambda/(2*m) * (sum(sum(Theta1(:,2:end).^2)) + sum(sum(Theta2(:,2:end).^2)));

% Use back propagation to compute the gradients
for t = 1:m;
	% Perform a feed forward pass to compute the activations (z2, a2, z3, a3)
	a1 = X(t,:)';
	z2 = Theta1 * a1;
	a2 = [1; sigmoid(z2)];
	z3 = Theta2 * a2;
	a3 = sigmoid(z3);
	
	% Calculate the error terms for each layer with a backward pass.  
	yk(y(t)) = 1;												% convert label to logical vector
	delta3 = a3 - yk;											% the output layer
	delta2 = Theta2(:,2:end)' * delta3 .* sigmoidGradient(z2);	% the hidden layer

	% Now accumulate the gradients
	Theta1_grad = Theta1_grad + delta2 * a1';
	Theta2_grad = Theta2_grad + delta3 * a2';
	
	yk(y(t)) = 0; 	% reinitialize the output vector
end

% Return the non-regularized gradients which are the partial derivatives of the cost function wrt Theta1 & Theta2
Theta1_grad = 1/m * Theta1_grad;
Theta2_grad = 1/m * Theta2_grad;

% Add regularization to the gradients.  Don't add to the bias column.
Theta1_grad(:,2:end) = Theta1_grad(:,2:end) + lambda/m * Theta1(:,2:end);
Theta2_grad(:,2:end) = Theta2_grad(:,2:end) + lambda/m * Theta2(:,2:end);

%-------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
