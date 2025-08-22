%% Polynomial Zonotope Approach for Jacobian Bounds
% Building on your existing code, add this section after loading the network

fprintf('\n===== POLYNOMIAL ZONOTOPE APPROACH =====\n');

% First, identify uncertain neurons in each layer (those in the ReLU's uncertain region)
% For demonstration, we'll assume some neurons are uncertain based on typical behavior

% Create matrix zonotopes for Delta_phi (activation subgradients)
matZono_list = cell(n_layers-1, 1);

% Track the number of generators (uncertain neurons) per layer
n_generators_per_layer = zeros(n_layers-1, 1);

for k = 1:(n_layers-1)
    activation_dim = size(weights_raw{k}, 1);
    
    % Center matrix for Delta_phi: 0.5 * I (middle of [0,1] interval)
    Delta_center = 0.5 * eye(activation_dim);
    
    % Generator matrices for uncertain neurons
    % Each uncertain neuron contributes one generator
    % For now, assume 30% of neurons are uncertain (you can adjust based on actual data)
    n_uncertain = round(0.3 * activation_dim);
    uncertain_indices = randperm(activation_dim, n_uncertain);
    
    n_generators_per_layer(k) = n_uncertain;
    
    % Create generator matrices
    Delta_generators = zeros(activation_dim, activation_dim, n_uncertain);
    
    for g = 1:n_uncertain
        idx = uncertain_indices(g);
        gen_matrix = zeros(activation_dim, activation_dim);
        % Radius is 0.5 for [0,1] interval
        gen_matrix(idx, idx) = 0.5;
        Delta_generators(:,:,g) = gen_matrix;
    end
    
    % Create matrix zonotope for this layer's Delta_phi
    matZono_list{k} = matZonotope(Delta_center, Delta_generators);
    
    fprintf('Layer %d: %d uncertain neurons out of %d\n', k, n_uncertain, activation_dim);
end

%% Compute Jacobian using Polynomial Zonotopes

% Start with the first layer: matrix zonotope × weight matrix → matrix zonotope
weight_1 = weights_raw{1}';  % Transpose for correct dimensions
J_matZono = matZono_list{1} * weight_1;

fprintf('\nBuilding Jacobian through layers:\n');
fprintf('After layer 1: Matrix Zonotope with %d generators\n', size(J_matZono.G, 3));

% For subsequent layers, we need to handle polynomial zonotopes
if n_layers > 2
    for layer = 2:(n_layers-1)
        % Current layer's Delta_phi as matrix zonotope
        Delta_current = matZono_list{layer};
        
        % Weight matrix for current layer
        W_current = weights_raw{layer}';
        
        % Compute Delta_phi * W for current layer
        DW_current = Delta_current * W_current;
        
        % Extract columns of J_matZono (previous Jacobian)
        [n_rows, n_cols] = size(J_matZono.C);
        
        % Initialize polynomial zonotope accumulator
        J_polyZono_accumulated = [];
        
        % Process each output dimension
        for col = 1:n_cols
            % Extract column as polynomial zonotope
            col_center = J_matZono.C(:, col);
            
            % Extract generators for this column
            n_gen = size(J_matZono.G, 3);
            col_generators = zeros(n_rows, n_gen);
            for g = 1:n_gen
                col_generators(:, g) = J_matZono.G(:, col, g);
            end
            
            % Create polynomial zonotope for this column
            % Using dependent generators (G) with identity exponent matrix
            expMat = eye(n_gen);  % Each generator has its own factor
            col_polyZono = polyZonotope(col_center, col_generators, [], expMat);
            
            % Multiply: polynomial zonotope × matrix zonotope
            % This captures dependencies across layers
            J_col_new = DW_current * col_polyZono;
            
            % Accumulate results
            if isempty(J_polyZono_accumulated)
                J_polyZono_accumulated = J_col_new;
            else
                % Combine polynomial zonotopes (stack columns)
                J_polyZono_accumulated = cartProd(J_polyZono_accumulated, J_col_new);
            end
        end
        
        % Convert back to matrix form for next iteration if needed
        % For the last layer before output, keep as polynomial zonotope
        if layer < (n_layers-1)
            % Convert polynomial zonotope back to matrix zonotope for next iteration
            J_matZono = matZonotope(J_polyZono_accumulated);
            fprintf('After layer %d: Polynomial Zonotope converted to Matrix Zonotope\n', layer);
        else
            % Keep as polynomial zonotope for final result
            J_polyZono_final = J_polyZono_accumulated;
            fprintf('After layer %d: Final Polynomial Zonotope\n', layer);
        end
    end
end

% Final multiplication with output layer weights
weight_output = weights_raw{n_layers}';
if exist('J_polyZono_final', 'var')
    % Multiple hidden layers case
    J_polyZono = weight_output * J_polyZono_final;
else
    % Single hidden layer case
    J_polyZono = weight_output * J_matZono;
end

%% Convert to interval matrix for norm computation
fprintf('\nConverting Polynomial Zonotope to Interval Matrix...\n');

% Convert polynomial zonotope to interval matrix
if isa(J_polyZono, 'polyZonotope')
    % Over-approximate polynomial zonotope as interval
    J_interval_from_polyZono = interval(J_polyZono);
    J_intervalMatrix_polyZono = intervalMatrix(J_interval_from_polyZono);
else
    % If it's still a matrix zonotope
    J_intervalMatrix_polyZono = intervalMatrix(J_polyZono);
end

%% Compute norms using polynomial zonotope approach
L1_norm_polyZono = norm(J_intervalMatrix_polyZono, 1);
Linf_norm_polyZono = norm(J_intervalMatrix_polyZono, Inf);

% Also compute the absolute upper bound
J_upper_bound_polyZono = abs(J_intervalMatrix_polyZono);
J_upper_bound_max_polyZono = max(J_upper_bound_polyZono, [], 'all');

%% Compare results
fprintf('\n===== COMPARISON OF METHODS =====\n');
fprintf('Interval Matrix Approach:\n');
fprintf('  L1 norm:   %.12f\n', L1_norm_cora);
fprintf('  L∞ norm:   %.12f\n', Linf_norm_cora);
fprintf('  Max bound: %.12f\n', J_upper_bound_max);

fprintf('\nPolynomial Zonotope Approach:\n');
fprintf('  L1 norm:   %.12f\n', L1_norm_polyZono);
fprintf('  L∞ norm:   %.12f\n', Linf_norm_polyZono);
fprintf('  Max bound: %.12f\n', J_upper_bound_max_polyZono);

fprintf('\nNaive Bounds:\n');
fprintf('  L1 norm:   %.12f\n', naive_upper_l1);
fprintf('  L∞ norm:   %.12f\n', naive_upper_linf);

% Calculate improvement ratios
improvement_L1 = (L1_norm_cora - L1_norm_polyZono) / L1_norm_cora * 100;
improvement_Linf = (Linf_norm_cora - Linf_norm_polyZono) / Linf_norm_cora * 100;

fprintf('\nImprovement of Polynomial Zonotope over Interval Matrix:\n');
fprintf('  L1 norm improvement: %.2f%%\n', improvement_L1);
fprintf('  L∞ norm improvement: %.2f%%\n', improvement_Linf);

%% Visualization of bounds (optional)
if n_layers == 4  % For 4-layer network
    figure;
    
    methods = categorical({'Naive L1', 'Naive L∞', 'Interval L1', 'Interval L∞', 'PolyZono L1', 'PolyZono L∞'});
    bounds = [naive_upper_l1, naive_upper_linf, L1_norm_cora, Linf_norm_cora, L1_norm_polyZono, Linf_norm_polyZono];
    
    bar(methods, bounds);
    ylabel('Lipschitz Bound');
    title('Comparison of Lipschitz Bounds');
    grid on;
    
    % Add values on top of bars
    text(1:length(bounds), bounds, num2str(bounds', '%.4f'), ...
        'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom');
end