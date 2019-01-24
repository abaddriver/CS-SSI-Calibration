function [csIm] = StackChannels(inIm)
% this function takes a spectral image, transposes each of its color
% channels and stacks the channels. this is done in order to convolve using
% a matrix.
% inputs:
%   inIm -  an image with multiple color layers.
% outputs:
%   csIm -  an image that has blocks of transposed color channels from
%           inIm.


% sizes:
assert(numel(size(inIm)) == 3);
[NXy, NXx, L] = size(inIm);

% transpose the cube:

csIm = zeros(NXx, NXy, L);
for ii=1:size(inIm,3)
    csIm(:,:,ii) = inIm(:,:,ii)';
end


% columnstack the result:
csIm = reshape(permute(csIm, [1,3,2]), ...
    [NXx*size(inIm,3) NXy]);

end