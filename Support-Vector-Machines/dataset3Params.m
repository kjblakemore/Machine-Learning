function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

parameters = [0.01; 0.03; 0.1; 0.3; 1.0; 3.0; 10.0; 30.0];	% possible C & sigma values

num = size(parameters,1);
errors = zeros(num^2,1);

% Train the SVM for all C/sigma pairs and compute the error for the 
% cross validation set.
for i = 1:num			% first vary C
	for j = 1:num		% then vary sigma
		model= svmTrain(X, y, parameters(i), @(x1, x2) gaussianKernel(x1, x2, parameters(j)));
		predictions = svmPredict(model, Xval);
		errors((i-1)*num+j) = mean(double(predictions != yval));
	end
end
		
% Find the minimum error and return the corresponding C & sigma.
% Use integer division and modulus to convert error index to parameters
% index.  Since Octave matrices are 1 based, need to first subtract one,
% then add one after floor and mod operations.
[val, index] = min(errors);
C = parameters(floor((index-1)/num)+1);
sigma = parameters(mod(index-1,num)+1);

% =========================================================================

end
