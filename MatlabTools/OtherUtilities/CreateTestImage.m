clear all; clc;

% create the test image:
filename = 'C:\Users\Amir\Documents\CS SSI\test\test.rawCube';
mypath1 = 'C:\Users\Amir\Desktop\225168_10150181183607103_3695919_n.jpg';
mypath2 = 'C:\Users\Amir\Desktop\IMG_20150305_150407.jpg';

myx1 = single(rgb2gray(imread(mypath1)));
myx2 = single(rgb2gray(imread(mypath2)));
mysz2 = size(myx2)/2;
myx2 = myx2(mysz2(1)-100:mysz2(1)+99,mysz2(2)-100:mysz2(2)+99);
myy = zeros(200,200,2,'single');
myy(:,:,1) = myx1;
myy(:,:,2) = myx2;
Im = myy;
type = 'single';
fileID = fopen(filename, 'wb');
fwrite(fileID, [size(Im,1) size(Im,2), size(Im,3)], 'int');
Im = permute(Im, [3 2 1]);
typestr = zeros(1,10, 'uint8');
typestr(1:numel(type)) = type;
fwrite(fileID, typestr, 'uint8');
fwrite(fileID, Im(:), 'single');
fclose(fileID);