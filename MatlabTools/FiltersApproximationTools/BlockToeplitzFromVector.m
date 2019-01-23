function [A] = BlockToeplitzFromVector(a, NAy, NAx, L, Nzeros)
% BlockToeplitzFromVector
% -----------------------
% this function returna a matrix with L blocks along the x dimension,
% where each block is a toeplitz matrix.
% inputs:
%   a - vector of the values to put it A_toep.
%   NAy - numer of rows in A
%   NAx - number of columns in A
%   L - number of blocks in A
%   Nzeros - number of values to be ommitted from beginning and end of each
%            block.

if nargin < 5
    Nzeros = [0 0];
end

assert(numel(Nzeros) == 2);

% sizes:
NBlock_x = NAx / L;
Na_block = NAy + NBlock_x - 1;

% allocate data
A = zeros(NAy, NAx);

% sanity checks
assert(mod(NAx,L) == 0)
assert(numel(a) + sum(Nzeros)*L == Na_block*L)

% all indices array:
j_inds = 1:NBlock_x;

% iterate over L  blocks:
for ll=1:L
    % allocate Block:
    A_block = zeros(NAy, NBlock_x);
    
    for a_i_ind = 1:(NAy + NBlock_x - 1 - sum(Nzeros))
        a_ind = a_i_ind + (ll-1)*(Na_block - sum(Nzeros)); % ind in big array
        
        % diag number: ind_i - ind_j = diag
        % smallest is bottom leftmost
        % biggest is upper rightmost
        diag = a_i_ind + Nzeros(1) - NAy;
        
        % calculate diag indices:
        i_diag = j_inds - diag;
        i_diag_inds = (i_diag >= 1 & i_diag <= NAy);
        i_diag = i_diag(i_diag_inds);
        j_diag = j_inds(i_diag_inds);
        A_block_inds = sub2ind(size(A_block), i_diag, j_diag);
        
        % put a_i value in diag indices:
        A_block(A_block_inds) = a(a_ind);
    end
    
    % save block in A:
    block_start_x = (ll-1)*NBlock_x + 1;
    block_end_x = ll*NBlock_x;
    A(:, block_start_x:block_end_x) = A_block;
    
end
end