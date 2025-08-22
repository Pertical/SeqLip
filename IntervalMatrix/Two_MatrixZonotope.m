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





