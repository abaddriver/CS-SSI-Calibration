%% script_createLinearEquationDatabase
% use this script to create a database of M and N matrixes such that 
% for the columnstack of the filters a: M*a = N
clear all; clc;

%% paths and definitions:
inPath = 'C:\Users\Amir\Documents\CS SSI\ImageDatabase\Train';
outRootPath = 'C:\Users\Amir\Documents\CS SSI\ImageDatabase\LinearEquationMatrices';

% definitions:
NFilt = 301;

% create a new directory:
outPath = fullfile(outRootPath, ['NFilt_' num2str(NFilt)]);
if ~isfolder(outPath)
    mkdir(outPath);
end

%% get list of files in path:
listings = dir(inPath);
filenames = {listings.name};
DDmatches = cellfun(@(x)~isempty(x), strfind(filenames, '_DD.rawImage'));
DDfilenames = filenames(DDmatches);
dropChars = numel('_DD.rawImage');
filenames = cellfun(@(x) x(1:end-dropChars), DDfilenames, 'UniformOutput', 0);
Cubefilenames = cellfun(@(x) [x '_Cube.rawImage'], filenames, 'UniformOutput', 0);

%% iterate over files and save data:
for ii=1:numel(filenames)
    ddim = Load3DRawImage(fullfile(inPath, DDfilenames{ii}));
    cubeim = Load3DRawImage(fullfile(inPath, Cubefilenames{ii}));
    
    [M, N] = LinearEquationOfFilterFromExamples(cubeim, ddim, NFilt);
    
    Save3DRawImage(M, fullfile(outPath, [filenames{ii} '_M.rawImage']));
    Save3DRawImage(N, fullfile(outPath, [filenames{ii} '_N.rawImage']));
    
end