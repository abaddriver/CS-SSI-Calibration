%% Test_LinearEquationOfFilterFromExamples
% this test shows how to approximate filters using
% LinearEquationOfFilterFromExamples
clear all; clc;

%% test sizes:
Xsize = [4, 4, 2];
NFilt = 3;

%% generate data:
X = normrnd(0.0, 1.0, Xsize);
F = normrnd(0.0, 1.0, [Xsize(3), NFilt]);
Y = FilterCube(X, F);

%% use the utility:
[M, N] = LinearEquationOfFilterFromExamples(X,Y,NFilt);

%% estimate F:
a = M\N;
F_est = reshape(a, fliplr(size(F)))';