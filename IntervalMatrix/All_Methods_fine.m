
clc; clear; close all; 

num_Lipneurons = [10 20 30 40 50 60 70 80 90 100];
L2_Liplayer   = [1953.302, 5090.753, 6722.621, NaN, NaN, NaN, NaN, NaN, NaN, NaN];
L2_Lipneuron   = [1248.271, 3574.687, 4452.049, 5862.846, NaN, NaN, NaN, NaN, NaN, NaN];

file_path = 'IntervalMatrix/saved_weights/five_hidden_layers/';
mat_files = dir(fullfile(file_path, '*.mat'));

% Sort files
[~, sort_idx] = sort(arrayfun(@(x) str2double(regexp(x.name, 'mnist_weights_five_layers_(\d+)\.mat', 'tokens', 'once')), mat_files));
mat_files = mat_files(sort_idx);

% Initialize arrays
neurons_per_layer = zeros(length(mat_files), 1); 
naive_upper = zeros(length(mat_files), 1);
naive_lower = zeros(length(mat_files), 1);
L1_bounds = zeros(length(mat_files), 1);
Linf_bounds = zeros(length(mat_files), 1);
L2_fromL1Linf = zeros(length(mat_files), 1);

% Arrays to store converted LipSDP results
L1_Liplayer_arr = zeros(length(mat_files), 1);
Linf_Liplayer_arr = zeros(length(mat_files), 1);
L1_Lipneuron_arr = zeros(length(mat_files), 1);
Linf_Lipneuron_arr = zeros(length(mat_files), 1);

for file = 1:length(mat_files)
    file_name = mat_files(file).name;
    fprintf('Loading file: %s\n', file_name);
    
    tokens = regexp(file_name, 'mnist_weights_five_layers_(\d+)\.mat', 'tokens');
    neurons_per_layer(file) = str2double(tokens{1}{1});
    
    network_data = load(fullfile(file_path, file_name));
    
    % Pass the L2 values to the function
    L2_Liplayer_val = L2_Liplayer(file);    % Use () not {}
    L2_Lipneuron_val = L2_Lipneuron(file);  % Use () not {}
    
    % Call function with L2 values
    [naive_upper_val, naive_lower_val, L1_upper, Linf_upper, L2_converted, ...
     L1_Lipneuron, L1_Liplayer, Linf_Lipneuron, Linf_Liplayer] = ...
     interval_matrix(network_data, L2_Liplayer_val, L2_Lipneuron_val);
    
    % Store results
    naive_upper(file) = naive_upper_val;
    naive_lower(file) = naive_lower_val;
    L1_bounds(file) = L1_upper;
    Linf_bounds(file) = Linf_upper;
    L2_fromL1Linf(file) = L2_converted;
    
    % Store converted LipSDP results
    L1_Liplayer_arr(file) = L1_Liplayer;
    Linf_Liplayer_arr(file) = Linf_Liplayer;
    L1_Lipneuron_arr(file) = L1_Lipneuron;
    Linf_Lipneuron_arr(file) = Linf_Lipneuron;
end
% 
% Plot L1 norm 
figure; hold on; grid on;
plot(neurons_per_layer, L1_bounds, '-s', 'LineWidth', 2, 'DisplayName', 'L1');

% Also plot converted LipSDP results
plot(neurons_per_layer, L1_Liplayer_arr, '--o', 'LineWidth', 2, 'DisplayName', 'LipSDP Layer L1 (converted)');
plot(neurons_per_layer, L1_Lipneuron_arr, '--^', 'LineWidth', 2, 'DisplayName', 'LipSDP Neuron L1 (converted)');

xlim([10, 50]);
%change the ylim to exponential scale
set(gca, 'YScale', 'log');
xlabel('Neurons per Hidden Layer');
ylabel('Lipschitz Bound');
legend('Location', 'best');


%Plot L2 norm 
figure; hold on; grid on;
plot(neurons_per_layer, L2_fromL1Linf, '-s', 'LineWidth', 2, 'DisplayName', 'L2 (converted)');

% Also plot converted LipSDP results
plot(neurons_per_layer, L2_Liplayer, '--o', 'LineWidth', 2, 'DisplayName', 'LipSDP Layer L2 (LipSDP)');
plot(neurons_per_layer, L2_Lipneuron, '--^', 'LineWidth', 2, 'DisplayName', 'LipSDP Neuron L2 (LipSDP)');

xlim([10, 50]);
%change the ylim to exponential scale
set(gca, 'YScale', 'log');
xlabel('Neurons per Hidden Layer');
ylabel('Lipschitz Bound');
legend('Location', 'best');


%Plot Linf norm
figure; hold on; grid on;
plot(neurons_per_layer, Linf_bounds, '-d', 'LineWidth', 2, 'DisplayName', 'L∞');

% Also plot converted LipSDP results
plot(neurons_per_layer, Linf_Liplayer_arr, '--o', 'LineWidth', 2, 'DisplayName', 'LipSDP Layer L∞ (converted)');
plot(neurons_per_layer, Linf_Lipneuron_arr, '--^', 'LineWidth', 2, 'DisplayName', 'LipSDP Neuron L∞ (converted)');

xlim([10, 50]);
%change the ylim to exponential scale
set(gca, 'YScale', 'log');
xlabel('Neurons per Hidden Layer');
ylabel('Lipschitz Bound');
legend('Location', 'best');


% Updated interval_matrix function
function [naive_upper, naive_lower, L1_upper, Linf_upper, L2_converted, ...
          L1_Lipneuron, L1_Liplayer, Linf_Lipneuron, Linf_Liplayer] = ...
          interval_matrix(data, L2_Liplayer_val, L2_Lipneuron_val)
    
    weights = data.weights;
    biases = data.bias; 
    n_layers = numel(weights);
    
    % Store weight and bias interval matrices in cells
    W = cell(n_layers, 1);
    b = cell(n_layers, 1);
    
    % Precompute Delta_phi for all but last layer
    Delta_phi = cell(n_layers-1, 1);
    
    for i = 1:n_layers
        W{i} = weights{i};
        b{i} = biases{i};
        
        if i < n_layers
            activation_dim = size(weights{i}, 1);
            center_matrix = 0.5 * eye(activation_dim);
            radius_matrix = 0.5 * eye(activation_dim);
            Delta_phi{i} = intervalMatrix(center_matrix, radius_matrix);
        end
    end
    
    % Compute Jacobian interval matrix
    J_intervalMatrix = W{end};
    for i = n_layers-1:-1:1
        J_intervalMatrix = J_intervalMatrix * Delta_phi{i} * W{i};
    end
    
    % Naive upper bound (L2)
    naive_upper_L2 = 1;
    for i = 1:n_layers
        naive_upper_L2 = naive_upper_L2 * norm(W{i}, 2);
    end
    
    % Naive lower bound (L2)
    W_prod = W{1};
    for k = 2:n_layers
        W_prod = W{k} * W_prod;
    end
    naive_lower_L2 = norm(W_prod, 2);
    
    % CORA norms
    L1_norm_cora = norm(J_intervalMatrix, 1);
    Linf_norm_cora = norm(J_intervalMatrix, Inf);
    
    % Convert to L2
    L2_converted = convertToL2(L1_norm_cora, Linf_norm_cora, J_intervalMatrix);

    % disp(size(J_intervalMatrix)); 
    % Convert LipSDP L2 values to L1 and Linf
    if ~isnan(L2_Lipneuron_val)
        [L1_Lipneuron, Linf_Lipneuron] = convertFromL2(L2_Lipneuron_val, J_intervalMatrix);
    else
        L1_Lipneuron = NaN;
        Linf_Lipneuron = NaN;
    end
    
    if ~isnan(L2_Liplayer_val)
        [L1_Liplayer, Linf_Liplayer] = convertFromL2(L2_Liplayer_val, J_intervalMatrix);
    else
        L1_Liplayer = NaN;
        Linf_Liplayer = NaN;
    end
    
    % Return values
    naive_upper = naive_upper_L2;
    naive_lower = naive_lower_L2;
    L1_upper = L1_norm_cora;
    Linf_upper = Linf_norm_cora;
end

% Convert L1 or Linf to L2
function L2_bounds = convertToL2(L1_norm, Linf_norm, interval_Matrix)
    [m, n] = size(center(interval_Matrix));
    
    fromL1 = sqrt(n) * L1_norm; 
    fromLinf = sqrt(m) * Linf_norm; 
    
    L2_bounds = min([fromL1, fromLinf]); 
end

% Convert FROM L2 to L1 and Linf
function [L1_upper, Linf_upper] = convertFromL2(L2_norm, J_intervalMatrix)
    [m, n] = size(center(J_intervalMatrix));
    
    L1_upper = sqrt(m) * L2_norm;    % L1 ≤ √m * L2
    Linf_upper = sqrt(n) * L2_norm;  % L∞ ≤ √n * L2
end

% clc; clear; close all; 

% % ----------------------------------------------------------------
% %         Layer (type)               Output Shape         Param #
% % ================================================================
% %             Linear-1               [-1, 1, 100]          78,500
% %               ReLU-2               [-1, 1, 100]               0
% %             Linear-3               [-1, 1, 100]          10,100
% %               ReLU-4               [-1, 1, 100]               0
% %             Linear-5               [-1, 1, 100]          10,100
% %               ReLU-6               [-1, 1, 100]               0
% %             Linear-7               [-1, 1, 100]          10,100
% %               ReLU-8               [-1, 1, 100]               0
% %             Linear-9               [-1, 1, 100]          10,100
% %              ReLU-10               [-1, 1, 100]               0
% %            Linear-11                [-1, 1, 10]           1,010
% % ================================================================



% % ==== Summary of Results ====
% % mnist_weights_five_layers_70.mat [layer] → L = nan, time = 30.063s
% % mnist_weights_five_layers_70.mat [neuron] → L = nan, time = 41.121s
% % mnist_weights_five_layers_60.mat [layer] → L = nan, time = 26.126s
% % mnist_weights_five_layers_60.mat [neuron] → L = nan, time = 36.908s
% % mnist_weights_five_layers_10.mat [layer] → L = 1953.302, time = 13.224s
% % mnist_weights_five_layers_10.mat [neuron] → L = 1248.271, time = 16.406s
% % mnist_weights_five_layers_30.mat [layer] → L = 6722.621, time = 18.904s
% % mnist_weights_five_layers_30.mat [neuron] → L = 4452.049, time = 27.679s
% % mnist_weights_five_layers_100.mat [layer] → L = nan, time = 39.167s
% % mnist_weights_five_layers_100.mat [neuron] → L = nan, time = 67.409s
% % mnist_weights_five_layers_20.mat [layer] → L = 5090.753, time = 22.921s
% % mnist_weights_five_layers_20.mat [neuron] → L = 3574.687, time = 24.774s
% % mnist_weights_five_layers_50.mat [layer] → L = nan, time = 25.054s
% % mnist_weights_five_layers_50.mat [neuron] → L = nan, time = 33.778s
% % mnist_weights_five_layers_90.mat [layer] → L = nan, time = 41.204s
% % mnist_weights_five_layers_90.mat [neuron] → L = nan, time = 65.145s
% % mnist_weights_five_layers_80.mat [layer] → L = nan, time = 37.781s
% % mnist_weights_five_layers_80.mat [neuron] → L = nan, time = 54.172s
% % mnist_weights_five_layers_40.mat [layer] → L = nan, time = 20.697s
% % mnist_weights_five_layers_40.mat [neuron] → L = 5862.846, time = 37.082s
% %LipSDP results 
% %LipSDP_results = load('IntervalMatrix/saved_weights/LipSDP_FiveHiddenLayers_results.mat');
% % LipSDP_results = load('IntervalMatrix/LipSDP_FiveHiddenLayers_results.mat')

% % Load multiple layers from mat
% % multi_layer_network = load("IntervalMatrix/saved_weights/mnist_weights_more_layers.mat");

% %LipSDP results
% % Neurons per layer
% num_Lipneurons = [10 20 30 40 50 60 70 80 90 100];

% % Layer formulation results
% L2_Liplayer   = [1953.302, 5090.753, 6722.621,   NaN,   NaN,   NaN,   NaN,   NaN,   NaN,   NaN];
% time_Liplayer = [13.224,  22.921,  18.904, 20.697, 25.054, 26.126, 30.063, 37.781, 41.204, 39.167];

% % Neuron formulation results
% L2_Lipneuron   = [1248.271, 3574.687, 4452.049, 5862.846,   NaN,   NaN,   NaN,   NaN,   NaN,   NaN];
% time_Lipneuron = [16.406, 24.774, 27.679, 37.082, 33.778, 36.908, 41.121, 54.172, 65.145, 67.409];


% file_path = 'IntervalMatrix/saved_weights/five_hidden_layers/';
% mat_files = dir(fullfile(file_path, '*.mat'));

% %sort the file name base on the numbers.
% [~, sort_idx] = sort(arrayfun(@(x) str2double(regexp(x.name, 'mnist_weights_five_layers_(\d+)\.mat', 'tokens', 'once')), mat_files));
% mat_files = mat_files(sort_idx);

% neurons_per_layer = zeros(length(mat_files), 1); 
% naive_bounds = zeros(length(mat_files), 1);
% L1_bounds = zeros(length(mat_files), 1);
% Linf_bounds = zeros(length(mat_files), 1);

% L1_Liplayer = zeros(length(mat_files), 1);
% Linf_Liplayer = zeros(length(mat_files), 1);

% L1_Lipneuron = zeros(length(mat_files), 1);
% Linf_Lipneuron = zeros(length(mat_files), 1);

% L2_fromL1Linf = zeros(length(mat_files), 1);

% for file = 1:length(mat_files)
%     file_name = mat_files(file).name;
%     fprintf('Loading file: %s\n', file_name);

%     tokens = regexp(file_name, 'mnist_weights_five_layers_(\d+)\.mat', 'tokens');
%     neurons_per_layer(file) = str2double(tokens{1}{1});

%     network_data = load(fullfile(file_path, file_name));

%     [naive_upper, naive_lower, L1_upper, Linf_upper, L2_converted] = interval_matrix(network_data, file);

%     naive_upper(file) = naive_upper;
%     naive_lower(file) = naive_lower;
%     L1_bounds(file) = L1_upper;
%     Linf_bounds(file) = Linf_upper;

%     L2_fromL1Linf(file) = L2_converted; 

    

% end 

% %convert the LipL2 to L1
% % L1_bounds_Liplayer = convertFromL2(L2_Liplayer, size(J_intervalMatrix));

% %Plot
% figure; hold on; grid on;
% % plot(neurons_per_layer, naive_upper, '-o', 'LineWidth', 2);
% % plot(neurons_per_layer, naive_lower, '-x', 'LineWidth', 2);
% plot(neurons_per_layer, L1_bounds, '-s', 'LineWidth', 2);
% plot(neurons_per_layer, Linf_bounds, '-d', 'LineWidth', 2);


% xlabel('Neurons per Hidden Layer');
% ylabel('Lipschitz Bound');
% title('Scaling of Lipschitz Bounds vs Neurons');
% legend('CORA L1','CORA L∞');


% %test
% % interval_matrix(multi_layer_network);

% %Load the five layers
% function [naive_upper, naive_lower, L1_upper, Linf_upper, L2_converted, L1_Lipneuron, L1_Liplayer, Linf_Lipneuron, Linf_Liplayer] = interval_matrix(data, file)

%     weights = data.weights;
%     biases = data.bias; 
%     n_layers = numel(weights);

%     % Store weight and bias interval matrices in cells
%     W = cell(n_layers, 1);
%     b = cell(n_layers, 1);

%     % Precompute Delta_phi for all but last layer
%     Delta_phi = cell(n_layers-1, 1);

%     for i = 1:n_layers
%         % W{i} = intervalMatrix(weights{i});
%         % b{i} = intervalMatrix(biases{i});
%         W{i} = weights{i};
%         b{i} = biases{i};

%         if i < n_layers
%             activation_dim = size(weights{i}, 1);
%             center_matrix = 0.5 * eye(activation_dim);
%             radius_matrix = 0.5 * eye(activation_dim);
%             Delta_phi{i} = intervalMatrix(center_matrix, radius_matrix);
%         end
%     end

%     J_intervalMatrix = W{end};
%     for i = n_layers-1:-1:1
%         J_intervalMatrix = J_intervalMatrix * Delta_phi{i} * W{i};
%     end


%     J_upper_bound = abs(J_intervalMatrix);
%     J_upper_bound_max = max(J_upper_bound, [], 'all');

%     % Naive upper bound 
%     for i = 1:n_layers
%         if i == 1
%             naive_upper_L2 = norm(W{i}, 2);
%         else
%             naive_upper_L2 = naive_upper_L2* norm(W{i}, 2);
%         end
%     end

%     % Naive lower bound, all the weights time together before norm
%     % Naive_lower = norm(W2 * W1, 2);

%     % Multiply all weight matrices together for naive lower bound
%     W_prod = W{1};
%     for k = 2:n_layers
%         W_prod = W{k}*W_prod;
%     end
%     naive_lower_L2 = norm(W_prod, 2);


%     % CORA norms
%     L1_norm_cora  = norm(J_intervalMatrix, 1);
%     Linf_norm_cora = norm(J_intervalMatrix, Inf);

%     naive_upper = naive_upper_L2;
%     naive_lower = naive_lower_L2;
%     L1_upper = L1_norm_cora;
%     Linf_upper = Linf_norm_cora;

%     L2_converted = convertToL2(L1_upper, Linf_upper, J_intervalMatrix);
    
%     %Load the L2 norm from L2_Liplayer and L2_Lipneuron

%     % L2_Liplayer   = [1953.302, 5090.753, 6722.621,   NaN,   NaN,   NaN,   NaN,   NaN,   NaN,   NaN];
%     % time_Liplayer = [13.224,  22.921,  18.904, 20.697, 25.054, 26.126, 30.063, 37.781, 41.204, 39.167];

%     % % Neuron formulation results
%     % L2_Lipneuron   = [1248.271, 3574.687, 4452.049, 5862.846,   NaN,   NaN,   NaN,   NaN,   NaN,   NaN];
%     % time_Lipneuron = [16.406, 24.774, 27.679, 37.082, 33.778, 36.908, 41.121, 54.172, 65.145, 67.409];

%     L2_Lipneuron_file = L2_Lipneuron{file};
%     L2_Liplayer_file = L2_Liplayer{file};

%     [L1_Lipneuron, Linf_Lipneuron] = convertFromL2(L2_Lipneuron_file, J_intervalMatrix);
%     [L1_Liplayer, Linf_Liplayer] = convertFromL2(L2_Liplayer_file, J_intervalMatrix);


%     % -------- Print Results --------
%     % fprintf('Max element upper bound: %.6f\n', J_upper_bound_max);
%     % fprintf('Naive upper bound (L∞): %.6f\n', naive_upper_linf);
%     % fprintf('CORA norm (L1): %.6f\n', L1_norm_cora);
%     % fprintf('CORA norm (L∞): %.6f\n', Linf_norm_cora);

% end

% %Use norm equivalent to convert L1 or Linf norm to L2 norm
% function L2_bounds = convertToL2(L1_norm, Linf_norm, interval_Matrix)
%     [m, n] = size(center(interval_Matrix));

%     fromL1 = sqrt(n) * L1_norm; 
%     fromLinf = sqrt(m) * Linf_norm; 

%     L2_bounds = min([fromL1, fromLinf]); 
% end

% function [L1_upper, Linf_upper] = convertFromL2(L2_norm, J_intervalMatrix)
%     [m, n] = size(center(J_intervalMatrix));

%     L1_upper = sqrt(m) * L2_norm;    % L1 ≤ √m * L2
%     Linf_upper = sqrt(n) * L2_norm;  % L∞ ≤ √n * L2

% end

% % function Lq = norm_equivalent(Lp, p, q, input_dim, output_dim)

% %     % Lp: input norm
% %     % Lq: output norm
% %     % p, q: norm types(1, 2, Inf)


% %     n = input_dim; 
% %     m = output_dim;

% %     if p == q
% %         Lq = Lp; % No conversion needed
% %         return; 
% %     end 


% % end 

