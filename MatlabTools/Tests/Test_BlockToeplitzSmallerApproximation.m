%% Test_BlockToeplitzSmallerApproximation
% this test creates 
clear all;
clc;

%% sizes
L = 10; % color channels
NAy = 100; % Y size in the x dimension
NA_Block_x = 500;
Nzeros = [50 51];
Na_block = NAy + NA_Block_x - 1;
NAx = NA_Block_x * L;
NXy = NAx;
NXx = 100;

%% create Data:
% random filters:
a = rand((Na_block-sum(Nzeros))*L, 1);
% random X:
X = rand(NXy, NXx);
% convert filters to Matrix:
A = BlockToeplitzFromVector(a, NAy, NAx, L, Nzeros);
% create Y using A and X:
Y = A*X;

%% test BlockToeplitzSmallerApproximation
[A_approx, a_approx] = BlockToeplitzSmallerApproximation(X, Y, L, Nzeros);