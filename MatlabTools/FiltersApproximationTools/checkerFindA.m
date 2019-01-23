clear all; clc;
load 'checkerCalibAXY';
%% define sizes:
num_block_elements_total = size(A) ./ [1 32];
num_block_elements_approx = [301 301];
Nzeros = num_block_elements_total - num_block_elements_approx;
%% find A_toep from X and Y:
[A_toep, a_toep] = BlockToeplitzSmallerApproximation(X,Y,L,Nzeros);

%% check error
error_old = Y - A*X;
error_old = sum(error_old(:).^2)/numel(Y);
error_new= Y - A_toep*X;
error_new = sum(error_new(:).^2)/numel(Y);