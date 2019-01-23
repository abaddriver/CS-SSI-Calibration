function [filters] = BlockToeplitzToFilters(Atop, L, NFilt)
% function BlockToeplitzToFilters
% -------------------------------
% this function converts a block toeplitz matrix into a vector of filters.
% inputs:
%   Atop -      block toeplits matrix
%   L -         number of blocks in the matrix (same as number of filters)
%   NFilt -     number of filter coefficients for each filter of the output.
% outputs:
%   filters -   output filters that correspond to the blockwise toeplitz
%               matrix.

NAy = size(Atop, 1);
NAx = size(Atop,2);
NABlock_x = NAx / L;
NFilt_half = (NFilt+1)/2;

%% sanity checks:
% A is devided to L blocks
assert(mod(NAx, L) == 0);
% there are enugh coefficients:
assert(NAy >= NFilt_half);
% make sure that NFilt is odd:
assert(mod(NFilt, 2) == 1);

%% allocate output:
filters = zeros(L, NFilt);

%% loop over blocks/filters:
for ll=1:L
    A_block = Atop(:, (ll-1)*NABlock_x + 1: ll*NABlock_x);
    filters(ll, 1: NFilt_half) =  A_block(NFilt_half:-1:1 , 1);
    filters(ll, NFilt_half:end) = A_block(1, 1:NFilt_half);
end

end