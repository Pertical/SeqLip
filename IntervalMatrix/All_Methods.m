clc; clear; close all; 

%Example network structure: 
% ----------------------------------------------------------------
%         Layer (type)               Output Shape         Param #
% ================================================================
%             Linear-1               [-1, 1, 500]         392,500
%               ReLU-2               [-1, 1, 500]               0
%             Linear-3               [-1, 1, 200]         100,200
%               ReLU-4               [-1, 1, 200]               0
%             Linear-5                [-1, 1, 50]          10,050
%               ReLU-6                [-1, 1, 50]               0
%             Linear-7                [-1, 1, 10]             510
% ================================================================
% Total params: 503,260
% Trainable params: 503,260
% Non-trainable params: 0

% Activation function is ReLU
% INPUT_SIZE = 784, OUTPUT_SIZE = 10 with one hidden layer of 50 neurons

%LipSDP results
% LipSDP % python solve_sdp.py --form layer --weight-path examples/saved_weights/mnist_weights_more_layers.mat
% LipSDP-Layer gives a Lipschitz constant of 2418.267
% Total time: 103.56 seconds
% LipSDP % python solve_sdp.py --form neuron --weight-path examples/saved_weights/mnist_weights_more_layers.mat
% LipSDP-Neuron gives a Lipschitz constant of 1897.363
% Total time: 811.36 seconds
% LipSDP % python solve_sdp.py --form network --weight-path examples/saved_weights/mnist_weights_more_layers.mat
% Error using sparse
% Requested 281627x78890765625 (587.8GB) array exceeds maximum array size preference (16.0GB). This might cause MATLAB to become unresponsive.



% Load multiple layers from mat
% multi_layer_network = load("IntervalMatrix/saved_weights/mnist_weights_more_layers.mat");
multi_layer_network = load("IntervalMatrix/saved_weights/five_hidden_layers/mnist_weights_five_layers_10.mat");

%Load the five layers


weights_raw = multi_layer_network.weights; 
biases_raw = multi_layer_network.bias; 

% Determine how many layers the network has
n_layers = length(weights_raw);

hidden_layer_idx = 1; % Counter for hidden layers
activation_layer_idx = 1; % Counter for activation layers

W = cell(n_layers, 1);
b = cell(n_layers, 1);

for i = 1:n_layers
    % Create variable names for weights and biases
    weight_name = sprintf('weight_%d', i);
    bias_name = sprintf('bias_%d', i);

    % Assign the value to the dynamically named variables in the workspace
    assignin('base', weight_name, intervalMatrix((weights_raw{i})));
    assignin('base', bias_name, intervalMatrix(biases_raw{i}));

    if i == 1
        input_dim = size(weights_raw{i}, 2);
    elseif i == n_layers
        output_dim = size(weights_raw{i}, 1);
    else
        % Hidden layer
        hidden_name = sprintf('hidden_%d', hidden_layer_idx);
        hidden_size = size(weights_raw{i}, 1);
        assignin('base', hidden_name, hidden_size);

        hidden_bias_name = sprintf('hidden_bias_%d', hidden_layer_idx);
        assignin('base', hidden_bias_name, intervalMatrix(biases_raw{i}));

        hidden_layer_idx = hidden_layer_idx + 1;
    end

    for k = 1:n_layers - 1
        % activation_name = sprintf('activation_%d', k);
        activation_dim = size(weights_raw{k}, 1);
        % assignin('base', activation_name, 'ReLU');

        Delta_phi_name = sprintf('Delta_phi_%d', k);

        center_matrix = 0.5 * eye(activation_dim);  % Center at 0.5
        radius_matrix = 0.5 * eye(activation_dim);  % Radius of 0.5 gives [0,1]
        delta_phi = intervalMatrix(center_matrix, radius_matrix);

        assignin('base', Delta_phi_name, delta_phi);

    end
end

clear hidden_layer_idx hidden_bias_name hidden_name weight_name bias_name Delta_phi_name; 

J_intervalMatrix_temp = [];
for i = 1:(n_layers) %max is 4

    weight_name = sprintf('weight_%d', i); 
    %Create an empty matrix for jacobian 

    if i < n_layers %max is 3

        Delta_phi_name = sprintf('Delta_phi_%d', i);

        Delta_phi_name = eval(Delta_phi_name);
        weight_name = eval(weight_name);

        if i == 1 
            J_intervalMatrix_temp = Delta_phi_name * weight_name; 

        else
            J_intervalMatrix_temp = Delta_phi_name * weight_name * J_intervalMatrix_temp; 
        end 

    else
        % fprintf(weight_name);
        weight_name = eval(weight_name);
        J_intervalMatrix = weight_name * J_intervalMatrix_temp; 
    end 

end

%Check if the loops above is correct
if n_layers == 4
    J_intervalMatrix_test = weight_4 * Delta_phi_3 * weight_3 * Delta_phi_2 * weight_2 * Delta_phi_1 * weight_1;

    % Check if they're approximately equal
    tolerance = 1e-10;
    diff_center = norm(center(J_intervalMatrix) - center(J_intervalMatrix_test), 'fro');
    diff_radius = norm(rad(J_intervalMatrix) - rad(J_intervalMatrix_test), 'fro');

    if diff_center < tolerance && diff_radius < tolerance
        fprintf('Loop implementation matches direct computation (within tolerance)\n');
        fprintf('Center difference: %e\n', diff_center);
        fprintf('Radius difference: %e\n', diff_radius);
    else
        fprintf('Mismatch between loop and direct computation\n');
        fprintf('Center difference: %e\n', diff_center);
        fprintf('Radius difference: %e\n', diff_radius);
    end
end

%Save all the computed results to mat
% save('workspace_backup.mat')
% load('workspace_backup.mat');

% abs - returns the absolute value bound of an interval matrix
J_upper_bound = abs(J_intervalMatrix); %returns a numeric matrix of absolute upper bounds
J_upper_bound_max = max(J_upper_bound, [], 'all');

%The navie upper bound in terms of L1
for i = 1:n_layers
    weight_name = sprintf('weight_%d', i);
    weight_name = eval(weight_name);

    if i == 1
        naive_upper_l1 = norm(weight_name, 1);
    else
        naive_upper_l1 = naive_upper_l1 * norm(weight_name, 1);
    end
end

%The navie upper bound in terms of L_inf
for i = 1:n_layers
    weight_name = sprintf('weight_%d', i);
    weight_name = eval(weight_name);

    if i == 1
        naive_upper_linf = norm(weight_name, Inf);
    else
        naive_upper_linf = naive_upper_linf * norm(weight_name, Inf);
    end
end

%Norms for intervalMatrix using CORA norm
L1_norm_cora = norm(J_intervalMatrix, 1); % 1-norm
% L2_norm_cora = norm(J_intervalMatrix, 2); % 2-norm
Linf_norm_cora = norm(J_intervalMatrix, Inf); % ∞-nor

fprintf('Naive upper bound (L1): %.12f\n', naive_upper_l1);
fprintf('Naive upper bound (L∞): %.12f\n', naive_upper_linf);
fprintf('CORA norm (L1): %.12f\n', L1_norm_cora);
fprintf('CORA norm (L∞): %.12f\n', Linf_norm_cora);

%Matrix Zonotope Approach
fprintf('===== MATRIX ZONOTOPE APPROACH =====\n');

%This part is to create matrix zonotope for each activation layer. 
Delta_phi_matZono = cell(n_layers-1, 1);

for k = 1:n_layers-1

    activation_dim = size(weights_raw{k}, 1);
    Delta_center = 0.5*eye(activation_dim); % Center at 0.5 
    n_generators = activation_dim;
    Delta_generators = zeros(activation_dim, activation_dim, n_generators);

    for i = 1:n_generators
        gen_matrix = zeros(activation_dim, activation_dim);
        gen_matrix(i, i) = 0.5; % Radius of uncertainty
        Delta_generators(:, :, i) = gen_matrix;
    end

    %Delta_phi_matZono{k} = reduceMatrixZonotope(matZonotope(Delta_center, Delta_generators), 'pca', 1, []);
    Delta_phi_matZono{k} = matZonotope(Delta_center, Delta_generators);
    
end



J_manually = weights_raw{n_layers} * Delta_phi_matZono{n_layers-1} * weights_raw{n_layers-1} * Delta_phi_matZono{n_layers-2} * weights_raw{n_layers-2};

% J_manually_1 = reduce(J_manually, 'pca', 1, []); 
J_test1 = reduce(J_manually, 'pca', 0.5, []);
J_test2 = reduceMatrixZonotope(J_manually, 'pca', 0.5, []); 


temp_value = Delta_phi_matZono{n_layers-3} * weights_raw{n_layers-3};
% 
% temp_value = reduce(temp_value, 'pca',1, []);

J_test2 = J_test2 *temp_value; 

J_manually = J_manually * temp_value;


function matZ_reduced = reduceMatrixZonotope(matZ, method, order, options)

    [m, n] = size(matZ.C);
    
    Z = zonotope(matZ);

    Z_reduced = reduce(Z, method, order, options);

    c_reduced = reshape(center(Z_reduced), m, n);
    
    G_vectorized = generators(Z_reduced);
    num_generators = size(G_vectorized, 2);
    G_reduced = zeros(m, n, num_generators);
    
    for i = 1:num_generators
        G_reduced(:, :, i) = reshape(G_vectorized(:, i), m, n);
    end
    
    matZ_reduced = matZonotope(c_reduced, G_reduced);
end

% J_matZono = Delta_phi_matZono{1} * weights_raw{1};

% %The rest of the Jacobian 
% for i = 2:(n_layers-1)
%     % J_i = Delta_phi_i * W_i * J_{i-1}
%     % matrix zonotope × matrix × matrix zonotope → matrix zonotope
%     temp_matZono = Delta_phi_matZono{i} * weights_raw{i};
%     J_matZono = temp_matZono * J_matZono;

%     fprintf('  Processing layer %d: Current generators = %d\n', i, size(J_matZono.G, 3));
% end

% J_matZono = weights_raw{n_layers} * J_matZono;

% %L1 and Linf norm for the J_matZono
% L1_norm_matZono = norm(J_matZono, 1);
% Linf_norm_matZono = norm(J_matZono, Inf);

% fprintf('Matrix Zonotope Results:\n');
% fprintf('  L1 norm: %.12f\n', L1_norm_matZono);
% fprintf('  L∞ norm: %.12f\n', Linf_norm_matZono);


% fprintf('===== POLYNOMIAL ZONOTOPE APPROACH =====\n');
% J_matZono = Delta_phi_matZono{1} * weights_raw{1}; 

% if n_layers > 2 
%     for layer = 2:(n_layers-1)
%        Delta_current = Delta_phi_matZono{layer};
%        W_current = weights_raw{layer}';
        
%        DW_current = W_current * Delta_current ;

%        [n_rows, n_cols] = size(J_matZono.C);

%        J_polyZono_accumulated = [];

%        for col = 1:n_cols

%             col_center = J_matZono.C(:, col);

%             n_gen = size(J_matZono.G, 3); 
%             col_generators = zeros(n_rows, n_gen);

%             for g = 1:n_gen
%                 col_generators(:, g) = J_matZono.G(:, col, g);

%             end

%             expMat = eye(n_gen);  
%             col_polyZono = polyZonotope(col_center, col_generators, [], expMat);

%             J_col_new = DW_current * col_polyZono;

%             if isempty(J_polyZono_accumulated)
%                 J_polyZono_accumulated = J_col_new;
%             else
%                 % Combine polynomial zonotopes by stacking columns
%                 J_polyZono_accumulated = cartProd(J_polyZono_accumulated, J_col_new);
%             end
%         end

%         % Update J_matZono with the accumulated polynomial zonotope
%         J_matZono_poly = J_polyZono_accumulated;

%     end

% end 

% J_matZono = matZonotope(J_matZono_poly);
% weight_output = weights_raw{n_layers};

% J_polyZono = weight_output * J_matZono_poly;

% %L1 and Linf Norm
% L1_norm_polyZono = norm(J_polyZono, 1);
% Linf_norm_polyZono = norm(J_polyZono, Inf);

% fprintf('Polynomial Zonotope Results:\n');
% fprintf('  L1 norm: %.12f\n', L1_norm_polyZono);
% fprintf('  L∞ norm: %.12f\n', Linf_norm_polyZono);

% matrix zonotope × weight matrix → matrix zonotope

% %Network information
% input_dim = 784; % Input
% n_hidden = 50; % Number of hidden neurons in the first layer
% n_output = 10; % Number of output classes for MNIST


% %Load the saved weight from LipSDP
% weights = load("IntervalMatrix/saved_weights/mnist_weights_2.mat");
% weights = weights.weights;

% W1 = weights{1}; % First layer weights
% W2 = weights{2}; % Second layer weights

% %%%%
% fprintf('Interval Arithmetic Example:\n');
% %Interval Arithmetic Example
% W1_interval = interval(W1);
% W2_interval = interval(W2);

% %Input x is between 0 and 1 for each pixel in the MNIST image
% x_lower = zeros(input_dim, 1);
% x_upper = ones(input_dim, 1);
% x_interval = interval(x_lower, x_upper); 

% %Forward pass through the network
% z1_interval = W1_interval * x_interval;
% %Apply ReLU
% h1_lower = max(0, infimum(z1_interval));
% h1_upper = max(0, supremum(z1_interval));
% h1_interval = interval(h1_lower, h1_upper);

% % Forward pass through the second layer
% z2_interval = W2_interval * h1_interval;
% output_lower = infimum(z2_interval);
% output_upper = supremum(z2_interval);
% output_interval = interval(output_lower, output_upper);

% %Activation function subgradient
% delta_phi1 = subgradient_relu(z1_interval);
% delta_phi1_interval = diag(delta_phi1);

% J_interval = W2_interval * delta_phi1_interval * W1_interval;
% J_intervalMatrix = intervalMatrix(J_interval);

% J_upper = supremum(abs(J_interval)); % Upper bound of the Jacobian

% Naive_upper = norm(W1, 2) * norm(W2, 2);
% % fprintf('Upper bound of the Jacobian: %.12f\n', J_upper);
% fprintf('Naive upper bound: %.12f\n', Naive_upper);

% Naive_lower = norm(W2 * W1, 2);
% fprintf('Naive lower bound: %.12f\n', Naive_lower);

% %Using the built-in norm function for intervals
% L1_norm_builtin = J_interval.norm(1);  % 1-norm
% Linf_norm_builtin = J_interval.norm(Inf); % ∞-norm
% L2_norm_builtin = norm(J_interval, 2); % 2-norm

% which norm

% fprintf('L1 norm of the Jacobian (builtin): %.12f\n', L1_norm_builtin);
% fprintf('L∞ norm of the Jacobian (builtin): %.12f\n', Linf_norm_builtin);
% fprintf('L2 norm of the Jacobian (builtin): %.12f\n', L2_norm_builtin); 



% %%%%%Matrix Zonotope Example
% fprintf('\nMatrix Zonotope Example:\n');

% uncertain_neurons = find(infimum(delta_phi1) < supremum(delta_phi1));
% n_uncertain = length(uncertain_neurons);

% %Inputs:
% %  C - center matrix (n x m)
% %  G - h generator matrices stored as (n x m x h)

% %Center Matrix
% Delta_center = diag(delta_phi1.center);

% %Generator matrices
% Delta_generators = zeros(n_hidden, n_hidden, n_uncertain);

% for k = 1:n_uncertain
%     idx = uncertain_neurons(k);
%     gen_matrix = zeros(n_hidden, n_hidden);
%     gen_matrix(idx, idx) = (supremum(delta_phi1(idx)) - infimum(delta_phi1(idx))) / 2;
%     Delta_generators(:,:,k) = gen_matrix;
% end

% Delta_matZono = matZonotope(Delta_center, Delta_generators);

% J_zono = W2 * Delta_matZono * W1;
% %Convert to interval matrix
% % J_intervalMatrix = intervalMatrix(J_zono);


% %%%%%Polynomial Zonotope Example
% fprintf('\nPolynomial Zonotope Example:\n');


% %%%%%Matrix Zonotope Example
% fprintf('\nSet Product Example:\n');

% %%
% %Input: 
% % - J_interval: interval matrix
% % - input interval: interval polytope/zonotope

% %Create X zonotope
% X_zono = zonotope(x_interval);

% x_center = X_zono.center;
% x_generators = X_zono.generators;

% J_lower = infimum(J_interval);
% J_upper = supremum(J_interval);

% J_intervalMatrix = intervalMatrix(J_interval);

% L2_bound = spb_jacobian(J_intervalMatrix);

% function L2_bound_spb = spb_jacobian(J_intervalMatrix)

%     n_samples = 2; 

%     J_lower = infimum(J_intervalMatrix);
%     J_upper = supremum(J_intervalMatrix);

%     [m, n] = size(J_lower); 

%     L2_norm_spb = zeros(n_samples, 1);

%     for i = 1:n_samples

%         vertex = rand(n, 1);

%         % disp(vertex); 
%         v_plus = max(vertex, 0);
%         v_minus = min(vertex, 0);

%         % disp(v_plus);
%         % disp(v_minus);

%         s_lower = J_lower * v_plus - J_upper * v_minus;
%         s_upper = J_upper * v_plus - J_lower * v_minus;

%         s_interval = interval(s_lower, s_upper);
%         % disp(s_interval);
%         s_abs_upper = supremum(abs(s_interval));

%         % disp(s_abs_upper);

%         L2_norm_spb(i) = norm(s_abs_upper, 2);
%     end

%     L2_bound_spb = max(L2_norm_spb);
%     fprintf('Set Product Bound: %.12f\n', L2_bound_spb);

% end


% function dg = subgradient_relu(z)
%     dg = interval(zeros(size(z)));
%     for i = 1:length(z)

%         if infimum(z(i)) > 0 %
%             dg(i) = interval(1,1);
%         elseif supremum(z(i)) < 0
%             dg(i) = interval(0,0);
%         else
%             dg(i) = interval(0,1);
%         end
%     end
    
% end



% %matPolytope to matZonotope is exact conversion
% %matPolytope to intervalMatrix is exact conversion

% %matZonotope to matPolytope is over-approximation conversion
% %matZonotope to intervalMatrix is exact conversion

% %intervalMatrix to matPolytope is over-approximation conversion
% %intervalMatrix to matZonotope is over-approximation conversion





