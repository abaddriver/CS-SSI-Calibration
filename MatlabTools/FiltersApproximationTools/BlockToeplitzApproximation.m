function [A, a] = BlockToeplitzApproximation(X, Y, L)
% function ToeplitzApproximation
% minimization of the followin term: (Y-AX)^2
% where A is a blockwise toeplitz matrix
% inputs:
%   Y - NYy x NYx image
%   X - Cube of  NXy x NXx
%   L - number of toeplitz blocks

% outpus:
%   a - different values of A in a row vector
%   A - The sensing blockwise toeplitz matrix


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

% M * a = N
% where a is a column stack of all a_i.
% dimensions:
% a is a column vector of size Na
% N is a column vector of size NN
% M is a matrix of sizes [NN Na]
N = zeros(NYy*NYx, 1, 'single');
M = zeros(NYy*NYx, Na, 'single');

%% fill M and N:
% iterating over Y's coordinates
for ii = 1:NYy
    block_start = NAy - ii + 1;
    block_end = block_start + Nblock_x - 1;
    
    % Fill N:
    N((ii-1)*NYx+1:ii*NYx) = Y(ii,:)';
    
    % Fill M:
    for jj=1:NYx
        M_row_ind = (ii-1)*NYx+jj;
        % iterating over blocks:
        M_ii = zeros(1, Na);
        for ll=1:L
            M_ii(block_start+(ll-1)*Na_i:block_end+(ll-1)*Na_i) = ...
                X((ll-1)*Nblock_x+1:ll*Nblock_x, jj);
        end
        M(M_row_ind, :) = M_ii;
    end
end

%% calculate A from linear equations:
a = M\N;
% turn a into a matrix:
A = BlockToeplitzFromVector(a, NAy, NAx, L);
end

