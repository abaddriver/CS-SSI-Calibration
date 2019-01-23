function [] = Save3DRawImage(Im, filename,type)
if nargin < 3
    type = 'single';
end

fileID = fopen(filename, 'wb');
fwrite(fileID, [size(Im,1) size(Im,2), size(Im,3)], 'int');
if numel(type) < 10
    typestr = zeros(1,10, 'uint8');
    typestr(1:numel(type)) = type;
    fwrite(fileID, typestr, 'uint8');
end
Im = permute(Im, [3 2 1]);
fwrite(fileID, Im(:), 'single');
fclose(fileID);
end