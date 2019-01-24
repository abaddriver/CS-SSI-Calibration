function [outData] =  getPythonCalibrationOutput(calibOutputDir)
% output graphs for all results in calibOutputDir

% get list of all sub directories in files
files = dir(calibOutputDir);
dirFlags = [files.isdir];
directories = files(dirFlags);
directories = directories(3:end);

outData = {};

% iterate over all sub directories:
for idir = 1:numel(directories)
    
    % parse directory name to get the folder charactaristics:
    dirname = directories(idir).name;
    [dirNFilt, dirweights, dirDDW, dirLoss, dirReg] = analyzeCalibration_ParseString(dirname);
    
    dirFiltsPath = fullfile(calibOutputDir, dirname, 'outputFilters.rawImage');
    dirFilts = Load3DRawImage(dirFiltsPath);
    
    outData{end+1}.name = dirname;
    outData{end}.Filters = dirFilts;
    outData{end}.NFilt = dirNFilt;
    outData{end}.weights = dirweights;
    outData{end}.DDW = dirDDW;
    outData{end}.loss = dirLoss;
    outData{end}.reg = dirReg;
end

end

function [NFilt, weights, DDW, loss, reg] = analyzeCalibration_ParseString(dirname)

% parse util and list of strings with name/value pairs:
getIdxFunc = @(List, strname) find(contains(List,strname));
getExactIdxFunc = @(List, strname) find(strcmp(List, strname));
splitList = strsplit(dirname, '_');

% get the values:
NFilt = str2num(splitList{getExactIdxFunc(splitList, 'NFilt')+1});
weights = splitList{getIdxFunc(splitList, 'weights')+1};
iDDW = getIdxFunc(splitList, 'DDW');
iloss = getIdxFunc(splitList, 'loss');
ireg = getIdxFunc(splitList, 'reg');
iregFactor = getIdxFunc(splitList, 'regFactor');

if isempty(iDDW)
    DDW = 256 + NFilt - 1;
else
    DDW = str2num(splitList{iDDW+1});
end

if isempty(iloss)
    loss = 'l2';
else
    loss = splitList{iloss-1};
end

if isempty(ireg)
    reg = 'None';
    regFactor = '';
else
    reg = splitList{ireg-1};
    if isempty(iregFactor)
        regFactor = ' 0.001';
    else
        regFactor = [' ' splitList{iregFactor+1}];
    end
end
reg = [reg regFactor];

end