clc; clear; close all;

% Example network structure:
% Linear-1: 784 -> 500 (ReLU)
% Linear-2: 500 -> 200 (ReLU)  
% Linear-3: 200 -> 50  (ReLU)
% Linear-4: 50  -> 10  (no activation)

% Load network
multi_layer_network = load("IntervalMatrix/saved_weights/mnist_weights_more_layers.mat");
weights_raw = multi_layer_network.weights;
biases_raw = multi_layer_network.bias;

n_layers = length(weights_raw);

% Convert to interval matrices and store in cell arrays
W = cell(n_layers, 1);
b = cell(n_layers, 1);
layer_dims = zeros(n_layers + 1, 1);

for i = 1:n_layers
    W{i} = intervalMatrix(weights_raw{i});
    b{i} = intervalMatrix(biases_raw{i});
    
    if i == 1
        layer_dims(1) = size(weights_raw{i}, 2);  % Input dimension
    end
    layer_dims(i+1) = size(weights_raw{i}, 1);     % Output dimension of layer i
end

input_dim = layer_dims(1);
output_dim = layer_dims(end);

fprintf('Network Architecture:\n');
fprintf('Input: %d\n', input_dim);
for i = 1:n_layers-1
    fprintf('Layer %d: %d neurons (ReLU)\n', i, layer_dims(i+1));
end
fprintf('Output: %d neurons (no activation)\n', output_dim);
fprintf('\n');

% ===== CREATE SUBDIFFERENTIAL MATRICES FOR RELU =====
% For global Lipschitz: use worst-case [0,1] for all neurons
Delta_phi = cell(n_layers - 1, 1);  % No activation after last layer

for k = 1:(n_layers - 1)
    activation_dim = layer_dims(k + 1);  % Number of neurons in layer k
    center_matrix = 0.5 * eye(activation_dim);
    radius_matrix = 0.5 * eye(activation_dim);
    Delta_phi{k} = intervalMatrix(center_matrix, radius_matrix);
end

% ===== COMPUTE JACOBIAN USING LOOP =====
J_intervalMatrix = W{1};  % Start with first weight matrix

% Build up the Jacobian: J = W_n * Δφ_{n-1} * W_{n-1} * ... * Δφ_1 * W_1
for i = 1:(n_layers - 1)
    J_intervalMatrix = Delta_phi{i} * J_intervalMatrix;
    if i < n_layers - 1
        J_intervalMatrix = W{i+1} * J_intervalMatrix;
    end
end
% Multiply by final layer weights (no activation after)
J_intervalMatrix = W{n_layers} * J_intervalMatrix;

% ===== VERIFICATION: COMPUTE JACOBIAN DIRECTLY =====
% For your 4-layer network: J = W4 * Δφ3 * W3 * Δφ2 * W2 * Δφ1 * W1
if n_layers == 4
    J_intervalMatrix_test = W{4} * Delta_phi{3} * W{3} * Delta_phi{2} * W{2} * Delta_phi{1} * W{1};
    
    % Check if they're equal
    % Check if they're approximately equal
    tolerance = 1e-10;
    diff_center = norm(center(J_intervalMatrix) - center(J_intervalMatrix_test), 'fro');
    diff_radius = norm(rad(J_intervalMatrix) - rad(J_intervalMatrix_test), 'fro');

    if diff_center < tolerance && diff_radius < tolerance
        fprintf('✓ Loop implementation matches direct computation (within tolerance)\n');
        fprintf('  Center difference: %e\n', diff_center);
        fprintf('  Radius difference: %e\n', diff_radius);
    else
        fprintf('✗ Mismatch between loop and direct computation\n');
        fprintf('  Center difference: %e\n', diff_center);
        fprintf('  Radius difference: %e\n', diff_radius);
    end
end

% ===== COMPUTE LIPSCHITZ CONSTANTS =====
L1_norm = norm(J_intervalMatrix, 1);
Linf_norm = norm(J_intervalMatrix, Inf);
L2_norm = norm(J_intervalMatrix, 2);

fprintf('===== GLOBAL LIPSCHITZ CONSTANTS =====\n');
fprintf('L1 Lipschitz constant: %.12f\n', L1_norm);
fprintf('L∞ Lipschitz constant: %.12f\n', Linf_norm);
fprintf('L2 Lipschitz constant: %.12f\n', L2_norm);

% ===== COMPARE WITH NAIVE BOUNDS =====
% Naive upper bound: product of all layer norms
L2_naive_upper = 1;
for i = 1:n_layers
    L2_naive_upper = L2_naive_upper * norm(weights_raw{i}, 2);
end
fprintf('\nNaive L2 upper bound (∏||W_i||): %.12f\n', L2_naive_upper);

% Lower bound: all ReLUs active (identity)
W_product = weights_raw{n_layers};
for i = (n_layers-1):-1:1
    W_product = W_product * weights_raw{i};
end
L2_lower_all_active = norm(W_product, 2);
fprintf('L2 lower bound (all ReLUs active): %.12f\n', L2_lower_all_active);

% Ratio to show tightness
fprintf('\nTightness ratio (L2_norm / L2_naive_upper): %.4f\n', L2_norm / L2_naive_upper);

% ===== OPTIONAL: Function for general n-layer networks =====
function J = compute_jacobian_interval(W, Delta_phi)
    % W: cell array of weight interval matrices
    % Delta_phi: cell array of subdifferential interval matrices
    
    n_layers = length(W);
    J = W{1};
    
    for i = 1:(n_layers - 1)
        J = Delta_phi{i} * J;
        if i < n_layers - 1
            J = W{i+1} * J;
        end
    end
    J = W{n_layers} * J;
end