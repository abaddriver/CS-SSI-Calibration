function [M,N] = LinearEquationOfFilterFromExamples(X, Y, NFilt)
% function LinearEquationOfFilterFromExamples
% this function create a matrix M such that if Y is the convolution result
% of X and F, then F is the solution of M*a = N where a is a columnstack of
% the filters that make F.
% inputs:
%   X -     the spectral cube, with dimensions (NXy, NXx, L)
%   Y -     the DD image, with dimensions (NYy, NYx)
%           if Y is bigger than the convolution size, it is cropped around
%           the center.
%   NFilt - the number of coefficients in each filter of F.
% outputs:
%   M -     output matrix M, of size (NYx*NYy, NFilt*L)
%   N -     output vector N, of size (NYx*NYy, 1)

%% sizes and dimensions:
NXy = size(X,1); NXx = size(X,2);
NYy = size(Y,1); NYx = size(Y,2);
L = size(X,3);
convsize = NXx + NFilt - 1;

% sanity checks:
assert(mod(NFilt,2) == 1)
assert(NYy == NXy);
assert(convsize <= NYx);

% crop Y to convsize:
if NYx ~= convsize
    xstart = (NYx-convsize)/2;
    xend = xstart + convsize - 1;
    Y = Y(:, xstart:xend);
    NYx = convsize;
end

%% fill the matrices with values from X:
M = zeros( NYy*NYx , NFilt*L);
Mrow = 1; % row index in M
for ii=1:NYy
    for jj=1:NYx
        
        % fill M:
        % X input indices:
        xstart=jj-NFilt+1; xstart = max(xstart,1);
        xend=jj; xend = min(xend, NXx);
        % M output indices:
        mstart=NFilt+1-jj; mstart=max(mstart,1);
        mend = mstart + xend - xstart;
        
        % fill in all L values:
        for ll=1:L
            M(Mrow, (mstart:mend) + (ll-1)*NFilt) = X(ii, xstart:xend, ll);
        end
               
        Mrow = Mrow + 1;
    end
end

% create the N vector:
N = reshape(Y.',[],1);
end