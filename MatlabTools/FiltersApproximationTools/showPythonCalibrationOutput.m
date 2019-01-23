function [outData] = showPythonCalibrationOutput(...
    calibOutputDirs, keepNFilts, keepweights, keeplosses, keepregs, ...
    Frequencies)

if nargin < 2
    keepNFilts = [];
end
if nargin < 3
    keepweights = {};
end
if nargin < 4
    keepweights = {};
end
if nargin < 5
    keeplosses = {};
end
if nargin < 6
    keepregs = {};
end

if nargin < 7
    Frequencies = [6, 15, 22];
end

%% get all the filters from the directories:
outDataOrig = {};
for ii=1:numel(calibOutputDirs)
    tmp = getPythonCalibrationOutput(calibOutputDirs{ii});
    outDataOrig(end+1:end+numel(tmp)) = tmp;
end
origFilters = [];

%% filter the data to match the inputs:
NFilts_filterfunc = @(x) any(x.NFilt == keepNFilts);
if ~isempty(keepNFilts)
    outData = outData(cellfun(NFilts_filterfunc, outData));
end

weights_filterfunc = @(x) any(strcmp(x.weights, keepweights));
if ~isempty(keepweights)
    outData = outData(cellfun(weights_filterfunc, outData));
end

loss_filterfunc = @(x) any(strcmp(x.loss, keeplosses));
if ~isempty(keeplosses)
    outData = outData(cellfun(loss_filterfunc, outData));
end

regs_filterfunc = @(x) any(strcmp(x.reg, keepregs));
if ~isempty(keepregs)
    outData = outData(cellfun(regs_filterfunc, outData));
end

%% show results:
%% show energy as function of filter across all outputs:
figure('units','normalized','outerposition',[0 0 1 1]);
subplot(4,1,1);
hold on;
wavelengths = 1:31;
filtLegends = {};
for ii=1:numel(outData)
    filters = outData{ii}.Filters;
    energies = sum(abs(filters), 2);
    plot(wavelengths, energies);
    filtLegends{end+1} = ['NFilt: ' num2str(outData{ii}.NFilt) ...
            ' weights: ' outData{ii}.weights ...
            ' loss: ' outData{ii}.loss ...
            ' reg: ' outData{ii}.reg];
end

if ~isempty(origFilters)
    energies = sum(abs(origFilters),2);
    plot(wavelengths, energies);
    filtLegends{end+1} = 'Original Filters';
end
title('energy as function of channel');
legend(filtLegends);
%% show selected wavelength across all outputs:
% selected frequencies:
plotii=2;

for gamma=Frequencies
    subplot(4,1,plotii);
    hold on;
    filtLegends = {};
    for ii=1:numel(outData)
        filter = outData{ii}.Filters(gamma,:);
        xx = (1:numel(filter)) - (numel(filter)+1)/2;
        plot(xx, filter);
        filtLegends{end+1} = ['NFilt: ' num2str(outData{ii}.NFilt) ...
            ' weights: ' outData{ii}.weights ...
            ' loss: ' outData{ii}.loss ...
            ' reg: ' outData{ii}.reg];
    end
    
    if ~isempty(origFilters)
        filter = origFilters(gamma,:);
        xx = (1:numel(filter)) - (numel(filter)+1)/2;
        plot(xx, filter);
        filtLegends{end+1} = 'Original Filter';
    end
    
    title(['filters for channel ' num2str(gamma)]);
    legend(filtLegends);
    plotii=plotii+1;
end
end