clc; clear; close all;
%Test script for Ahn & Chen's method to compute the exact L2-norm of an interval matrix.
%This comes from the paper's example 1

%interval
% Inputs:
%    I - interval object
%    a - lower limit
%    b - upper limit
%
% Outputs:
%    obj - generated interval object

%IntervalMatrix
% Inputs:
%    I - interval object
% Outputs:
%    intMat - intervalMatrix object


A_lower = [2 1;
           0 0;
           0 2];
       
A_upper = [3 1;
           2 1;
           1 3];

A_interval = interval(A_lower, A_upper);


% A_center = (A_upper + A_lower) / 2;
% A_radius = (A_upper - A_lower) / 2;

A_intervalMatrix = intervalMatrix(A_interval);

%Calculate the max singular value of the interval matrix
msv_test_chen = max_singular_value(A_intervalMatrix);
msv_test_matlab = norm(A_interval, 2 );
msv_intervalMatrix = norm(A_intervalMatrix);
fprintf('Max singular value of the test interval matrix: %.12f\n', msv_test_chen);
fprintf('L2 norm from Matlab Norm: %.12f\n', msv_test_matlab);
fprintf('Reference value (paper):%.12f\n', 4.54306177572459);


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
