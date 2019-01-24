function [Aest] = PythonCalibrationToBlockToeplitz(filters, NAy, NAx)
% function PythonCalibrationToBlockToeplitz
% -----------------------------------------
% this function converts a matrix whose rows are wavelength filters
% to a block toeplitz matrix that simulates the cross correlation.
% inputs:
%   filters -   a matrix that has L rows, each is the DD psf for the
%               wavelength assosiated with that channel.
%   NAy -       number of rows in the output matrix (typially 2592)
%   NAx -       number of columns in the output matrix (typically 256*L)

% get sizes from inputs:
L = size(filters,1); % number of channels
NFilt = size(filters,2); % number of coefficients in convolution

% size of matrix must divide appropriately with the number of channels:
assert(mod(NAx, L) == 0);
NBlock_x = NAx / L; % x size of each block matrix
conv_size = NBlock_x + NFilt - 1; % size of convolution
% convolution size must be smaller then the output:
assert(conv_size <= NAy);

% calculate the number of zeros for conv_size convolution:
NZeros_y = conv_size - NFilt;
NZeros_x = (conv_size + NBlock_x - 1 - NZeros_y - NFilt);
NZeros = [NZeros_y NZeros_x];

% calculate the difference for NAy convolution:
diff_size = (NAy - conv_size);
assert(mod(diff_size, 2) == 0);
NZeros = NZeros + diff_size/2;

% convert filters to a column vector:
filters = filters';
filters = filters(:);

% calculate convolution matrix:
Aest = BlockToeplitzFromVector(filters, NAy, NAx, L, NZeros);
end