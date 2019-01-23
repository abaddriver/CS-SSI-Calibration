function [filtIm, csIm] = FilterCubeWithMatrix(Cube, A)
% filter a cube with a blockwise toeplitz matrix.
% that is, sum of filters in all channels of Cube with the filter
% that is in the matching block of A.

% inputs:
%   Cube -      cube image
%   A -         blockwise toeplitz matrix
% output:
%   filtIm -    the filtered image
%   csIm -      the cube, transposed and stacked (color channel reduced to
%               in the y axis of the transpose).

% check for dimensions:
NXy = size(Cube,1); NXx = size(Cube,2); L = size(Cube,3);
NAy = size(A,1); NAx = size(A,2);
% verify compatibility:
assert(NAx == NXx * L);

% stack the channels to prepare for convolution using block toeplitz
% matrix:
csIm = StackChannels(Cube, true);

% filter:
filtIm = A*csIm;
filtIm = filtIm';
end