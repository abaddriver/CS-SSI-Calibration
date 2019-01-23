%% script_PythonCalibrationToBlockToeplitz_test
clc; clear all;
%% create a convolution matrix for NAx and NFilt where NFilt < NAx:
% sizes:
NAx = 20;
NFilt = 15;
L=1;
NAy = NAx + NFilt - 1;

% create random data:
x = normrnd(0,1,[NAx,1]);
filters = normrnd(0,1, [1,NFilt]);

% calculate sizes:
NZeros_y = NAy - NFilt;
NZeros_x = (NAy + NAx - 1 - NZeros_y - NFilt);
NZeros = [NZeros_y NZeros_x];

% calculate convolution matrix:
Aest = BlockToeplitzFromVector(filters, NAy, NAx, L, NZeros);

% convolve using matrix:
y1 = Aest * x;

% convolve using conv2:
y2 = conv2(x(:)', fliplr(filters(:)'));

% display difference:
sum(abs(y1(:) - y2(:)))


%% create a convolution matrix for NAx and NFilt where NAx > NFilt:

% sizes:
NAx = 10;
NFilt = 15;
L=1;
NAy = NAx + NFilt - 1;

% create random data:
x = normrnd(0,1,[NAx,1]);
filters = normrnd(0,1, [1,NFilt]);

% calculate zero sizes:
NZeros_y = NAy - NFilt;
NZeros_x = (NAy + NAx - 1 - NZeros_y - NFilt);
NZeros = [NZeros_y NZeros_x];

% calculate convolution matrix:
Aest = BlockToeplitzFromVector(filters, NAy, NAx, L, NZeros);

% convolve using matrix:
y1 = Aest * x;

% convolve using conv2:
y2 = conv2(x(:)', fliplr(filters(:)'));

% display difference:
sum(abs(y1(:) - y2(:)))

%% check the new script:

NAx = 20;
NFilt = 15;
L=1;
NAy = NAx + NFilt - 1;

AddN = 10;

% create random data:
x = normrnd(0,1,[NAx,1]);
filters = normrnd(0,1, [1,NFilt]);

% get matrix:
Aest = PythonCalibrationToBlockToeplitz(filters, NAy + 2*AddN, NAx);

% convolve using matrix:
y1 = Aest * x;

% convolve using conv2:
y2 = conv2(x(:)', fliplr(filters(:)'));

% display difference:
sum(abs(y1(AddN+1:end - AddN) - y2(:)))

