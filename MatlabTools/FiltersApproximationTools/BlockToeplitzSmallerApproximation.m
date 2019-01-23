function [A, a] = BlockToeplitzSmallerApproximation(X, Y, L, Nzeros)
% function BlockToeplitzSmallerApproximation
% minimization of the followin term: (Y-AX)^2
% where A is a blockwise toeplitz matrix
% inputs:
%   Y - NYy x NYx image
%   X - Cube of  NXy x NXx
%   L - number of toeplitz blocks
%   Nzeros - array of size 2x1 specifying the number of values to zero out
%            in x and y directions of each block of A.

% outpus:
%   a - different values of A in a row vector
%   A - The sensing blockwise toeplitz matrix

if nargin < 4
    Nzeros = [0 0];
end

% rearrange X and Y to create M and M such that N = M * a
% if Y=AX and A is blockwise toeplitz, and:
% A = | A1 A2 ... AL|
% each Ai is:
%        |a_i_0  a_i_1  a_i_2 ... |
%        |a_i_-1 a_i_0  a_i_1 ... |
%        |a_i_-2 a_i_-1 a_i_0 ... |
%         ...
%        |________________ ... |
%
% then:
% Y(i,j) = sum_k(A(i,k)*X(k,j)) = sum_l(sum_k(a_l_k-i * x(k+l*L,j)))
%
% assign:
% N: column vector, N(j) = Y(floor(j-1/SXx)+1, ((j-1)%SXx)+1) 1<=j<=SYx*SYy

%% sizes:
[NXy, NXx] = size(X);
[NYy, NYx] = size(Y);

assert(NYx == NXx);

% size of A matrix
NAx = NXy;
NAy = NYy;

% size of the a vector:
Nblock_x = NAx / L;
Na_i = Nblock_x + NAy - 1;
Na = Na_i * L;

% sanity check: number of zeros is not smaller then number of different
% coeffs in the toeplitz matrix
assert(Nzeros(1) < NAy);
assert(Nzeros(2) < Nblock_x);

% smaller number of vector block size:
Na_i_small = Na_i - sum(Nzeros);
Na_small = Na_i_small * L;

% M * a = N
% where a is a column stack of all a_i.
% dimensions:
% a is a column vector of size Na_small
% N is a column vector of size NN
% M is a matrix of sizes [NN Na_small]

% define number of equations:
NN = 2*Na_small;

N = zeros(NN, 1);
M = zeros(NN , Na_small);

%% fill M and N:
% select Na_small random inds from Y
Y_inds = randperm(numel(Y));
t = 1; % index in M to be filled
for kk = 1:numel(Y_inds)
    [ii,  jj] = ind2sub(size(Y), Y_inds(kk));
    
    % block toeplitz matrix a_i:
    a_i_start_orig = NAy - ii + 1; % first non zero index if Nzeros is [0 0]
    a_i_start = max(a_i_start_orig, Nzeros(1)+1); % first non zero index
    a_i_end_orig = a_i_start_orig + Nblock_x - 1; % last non zero index if Nzeros is [0 0]
    a_i_end = min(a_i_end_orig, Na_i - Nzeros(2)); % last non zero index
    
    % indices in X(:, jj):
    Xblock_start = a_i_start - a_i_start_orig + 1;
    Xblock_end = Xblock_start + a_i_end - a_i_start;
    
    % indices in M(kk, :):
    Mblock_start = a_i_start - Nzeros(1);
    Mblock_end = a_i_end - Nzeros(1);
    
    % Fill M:
    for ll=1:L
        M(t, (ll-1)*Na_i_small+Mblock_start : (ll-1)*Na_i_small+Mblock_end) = ...
            X((ll-1)*Nblock_x + Xblock_start: (ll-1)*Nblock_x + Xblock_end, jj);
    end
    
    % check if there is any relevant data in this row
    if sum(abs(M(t,:))) == 0
        continue;
    end
    
    % fill N:
    N(t) = Y(ii, jj);
    
    t = t + 1;
    if t > NN
        disp(['t is ' num2str(t) ', kk is ' num2str(kk) ', ' num2str(t/kk) ' precent']);
        break;        
    end    
end

%% calculate A from linear equations:
a = M\N;

% turn a into a matrix:
A = BlockToeplitzFromVector(a, NAy, NAx, L, Nzeros);
end