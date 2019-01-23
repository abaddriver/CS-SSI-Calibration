function [outs] = ParseLog(logPath, testfunc)

if nargin < 2
    testfunc = @getParameters;
end

%logPath = 'C:\Users\Amir\Documents\CS SSI\CalibrationOutputs\2018_11_01\2018-11-01__17-28-29.log'

fid = fopen(logPath);
tline = fgetl(fid);
outs = [];

while ischar(tline)
    if contains(tline, 'Iter:')
        tempVals = testfunc(tline);
        outs = [outs; tempVals];
    end
    tline = fgetl(fid);
end
end


function [outNumbers] = getParameters(Str)
key1 = 'ValidLoss:';
index1 = strfind(Str, key1);
validLoss = sscanf(Str(index1(1) + length(key1):end), '%g', 1);

key2 = 'TrainLoss:';
index2 = strfind(Str, key2);
trainLoss = sscanf(Str(index2(1) + length(key2):end), '%g', 1);

outNumbers = [validLoss trainLoss];
end