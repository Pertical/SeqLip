clc; clear; close all;

input_dim = 784; % Input dimension for MNIST images
n_hidden = 50; % Number of hidden neurons in the first layer
n_output = 10; % Number of output classes for MNIST

% Load the weight matrix 
weights = load("IntervalMatrix/saved_weights/mnist_weights_2.mat"); 
weights = weights.weights;

W1 = weights{1}; % First layer weights
W2 = weights{2}; % Second layer weights

%Minist input as Zonotope
x_lower = zeros(input_dim, 1); 
x_upper = ones(input_dim, 1);

x = interval(x_lower, x_upper);

z1 = W1 * x;

h1_lower = max(0, infimum(z1));
h1_upper = max(0, supremum(z1));
h1 = interval(h1_lower, h1_upper);

z2 = W2 * h1;
output = interval(z2);

%Activation function subgradient
delta_phi1 = subgradient_relu(z1);

delta_phi1 = intervalMatrix(delta_phi1);

delta_phi1_lower = infimum(delta_phi1);
delta_phi1_upper = supremum(delta_phi1);
% disp(delta_phi1_lower);
% disp(delta_phi1_upper);

delta_phi1_center = (delta_phi1_lower + delta_phi1_upper) / 2;
Delta_center = diag(delta_phi1_center);


%check how many neurons are in [0, 1]
%if its in [0, 1] then it is uncertain, if its in [1, 1] then its active 
%if its in [0, 0] then its inactive
uncertain_neurons = find(delta_phi1_lower < delta_phi1_upper);
n_uncertain = length(uncertain_neurons);

Delta_generators = zeros(n_hidden, n_hidden, n_uncertain);

for k = 1:n_uncertain
    idx = uncertain_neurons(k);
    gen_matrix = zeros(n_hidden, n_hidden);
    gen_matrix(idx, idx) = (delta_phi1_upper(idx) - delta_phi1_lower(idx)) / 2;
    Delta_generators(:,:,k) = gen_matrix;
end

Delta_matZono = matZonotope(Delta_center, Delta_generators);

J_zono = W2 * Delta_matZono * W1;

J_intervalMatrix = intervalMatrix(J_zono);

J_interval = interval(J_intervalMatrix);

l1_norm = norm(J_zono, 1);
linfi_norm = norm(J_zono, Inf);

%matPolytope to matZonotope is exact conversion
%matPolytope to intervalMatrix is exact conversion

%matZonotope to matPolytope is over-approximation conversion
%matZonotope to intervalMatrix is exact conversion

%intervalMatrix to matPolytope is over-approximation conversion
%intervalMatrix to matZonotope is over-approximation conversion

%Find the L2 norm for the zonotope
% l2_norm = norm(J_intervalMatrix, 2);

% l2_norm_chen = max_singular_value(J_intervalMatrix);
% 


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

%Ahn & Chen 's method for computing the exact L2-norm of a interval matrix.
function L2_exact = max_singular_value(J_interval)

    [m, n] = size(J_interval); 

    J_upper = supremum(J_interval);
    J_lower = infimum(J_interval);
    J_center = (J_upper + J_lower) / 2;
    J_radius = (J_upper - J_lower) / 2;

    total_patterns = 2^(m+n-1); 

    max_sv = 0;

    for pattern = 0:(total_patterns - 1)

        y_signs = ones(n, 1);
        
        % Fix y1 = 1 as the symmetry condition
        % y_signs(1) = 1; % y1 is always positive

        for i = 2:n
            if bitget(pattern, i-1)
                y_signs(i) = -1; % Set the sign of y_i based on the pattern
            end
        end

        z_signs = ones(m, 1);
        for j = 1:m
            if bitget(pattern, n + j - 1)
                z_signs(j) = -1; % Set the sign of z_j based on the pattern
            end
        end

        vertex_matrix = zeros(m, n);
        for i = 1:m
            for j = 1:n
                if y_signs(j) * z_signs(i) >= 0
                    vertex_matrix(i,j) = J_upper(i, j);
                else
                    vertex_matrix(i,j) = J_lower(i, j);
                end
            end
        end

        singular_value = svds(vertex_matrix, 1); % Compute the largest singular value
        max_sv = max(max_sv, singular_value);
    end 
    L2_exact = max_sv; 
end




