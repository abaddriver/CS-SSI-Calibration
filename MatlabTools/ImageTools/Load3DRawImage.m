function [outIm] = Load3DRawImage(filename)
fileID = fopen(filename, 'rb');
outSize = fread(fileID, [1 3], 'int');
outType = fread(fileID, 10,'*char')';
outIm = fread(fileID, Inf, 'single');
fclose(fileID);
outIm = reshape(outIm, fliplr(outSize));
outIm = permute(outIm, [3 2 1]);
end

