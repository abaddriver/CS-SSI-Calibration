function [imout] = FilterCube(Cube,Filt)
% this function filters a cube with a matrix of filters.
% inputs:
%   Cube -      image with L color channels.
%   Filt -      matrix of L filters, each one of length NFilt.
% outputs:
%   imout -     filtered image: sum of filtering of each color channel with
%               the matching filter.


% sizes:
% cube dimensions:
NXy = size(Cube,1); NXx = size(Cube,2); L = size(Cube,3);
% filter dimensiosns:
L1 = size(Filt, 1); NFilt = size(Filt,2);
% output dimensions:
conv_size = NXx + NFilt - 1;

% number of channels must be equal in Cube and in Filt:
assert(L == L1);

% run filtering:
imout = zeros(NXy, conv_size);
for ii=1:L
    imout = imout + ...
        conv2(1, fliplr(Filt(ii,:)), Cube(:,:,ii));
end

end

