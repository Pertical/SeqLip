clc; clear; close all; 


% ----------------------------------------------------------------
%         Layer (type)               Output Shape         Param #
% ================================================================
%             Linear-1                [-1, 1, 50]          39,250
%               ReLU-2                [-1, 1, 50]               0
%             Linear-3                [-1, 1, 10]             510
% ================================================================

% Information about the LipSDP Mnist example


% Load the weight Matrix VDBVFCfrom the LipSDP
weights = load("IntervalMatrix/saved_weights/mnist_weights_2.mat"); 

weights = weights.weights; 

W1 = weights{1}; % First layer weights
W2 = weights{2}; % Second layer weights

W1_interval = interval(W1);
W2_interval = interval(W2);

W1_dim = size(W1);
W2_dim = size(W2);

input_dim = 784;
n_hidden = 50;

%X interval is between 0 and 1
x_lower = zeros(input_dim, 1); 
x_upper = ones(input_dim, 1);

% Input interval would be 0 to 1 for each pixel in the MNIST image
x = interval(x_lower, x_upper);

% Forward pass through the first layer
z1 = W1_interval * x;

delta_phi1 = subgradient_relu(z1);

%Apply ReLU after the first layer
h1_lower = max(0, infimum(z1));
h1_upper = max(0, supremum(z1));
h1 = interval(h1_lower, h1_upper);

% Forward pass through the second layer
z2 = W2_interval * h1;

output_lower = infimum(z2);
output_upper = supremum(z2);
output_interval = interval(output_lower, output_upper);

dealt_phi1_lower = infimum(delta_phi1);
dealt_phi1_upper = supremum(delta_phi1);
% delta_phi1_interval = interval(diag(dealt_phi1_lower + dealt_phi1_upper));
delta_phi1_interval = diag(delta_phi1);
% 

J_interval = W2_interval * delta_phi1_interval * W1_interval;

J_upper = supremum(abs(J_interval)); % Upper bound of the Jacobian

%Using the built-in norm function for intervals
L1_norm_builtin = J_interval.norm(1);  % 1-norm
Linf_norm_builtin = J_interval.norm(Inf); % ∞-norm
% In the cora document B.3 Interval Matrices: norm - computes exactly the
% maximum norm value of all possible matrices. 
L2_norm_builtin = J_interval.norm(2); % 2-norm
L2_norm_cora = norm(J_interval, 2);

%L1 norm 
L1_norm = max(sum(abs(J_upper), 1));
fprintf('L1 norm of the Jacobian: %.12f\n', L1_norm);
fprintf('L1 norm of the Jacobian (builtin): %.12f\n', L1_norm_builtin);

% L∞ norm
Linf_norm = max(sum(abs(J_upper), 2));
fprintf('L∞ norm of the Jacobian: %.12f\n', Linf_norm);
fprintf('L∞ norm of the Jacobian (builtin): %.12f\n', Linf_norm_builtin);

%L2 norm using norm bounds
J_center = (J_upper + infimum(J_interval)) / 2;
J_radius = (J_upper - infimum(J_interval)) / 2;
L2_norm = norm(J_center, 'fro') + norm(J_radius, 'fro');
fprintf('L2 norm of the Jacobian (using bounds): %.12f\n', L2_norm);
fprintf('L2 norm of the Jacobian (builtin): %.12f\n', L2_norm_builtin); 

% L2 norm with ahn_chen_max_singular_value
% L2_exact = max_singular_value(J_interval);
% fprintf('Exact L2 norm of the Jacobian: %.12f\n', L2_exact);

L2_upper_naive = norm(W2, 2) * norm(W1, 2);
fprintf('Naive Upper Bound: %.12f\n', L2_upper_naive);

L2_lower_naive = norm(W2 * W1, 2);
fprintf('Naive Lower Bound: %.12f\n', L2_lower_naive);

% Test the matrix Zonotope

W1_interval_test = intervalMatrix(W1);
W1_zonotope = matZonotope(W1_interval_test); 

function dg = subgradient_relu(z)
    dg = interval(zeros(size(z)));
    for i = 1:length(z)
        if infimum(z(i)) > 0 %
            dg(i) = interval(1,1);
        elseif supremum(z(i)) < 0
            dg(i) = interval(0,0);
        else
            dg(i) = interval(0,1);
        end
    end
    
end
