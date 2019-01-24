%% get the data:
close all; clear all; clc;
calibOutputDirs = { ...
    'C:\Users\Amir\Documents\CS SSI\CalibrationOutputs\Est_2019-01-07_10-39-53'};

%'C:\Users\Amir\Documents\CS SSI\CalibrationOutputs\EstSensitivity_2019-01-03_15-08-45'};
% C:\Users\Amir\Documents\CS SSI\CalibrationOutputs\Est_2018-11-29_19-29-40_l2_loss
% C:\Users\Amir\Documents\CS SSI\CalibrationOutputs\Est_2018-12-12_12-52-54_l1_loss
% C:\Users\Amir\Documents\CS SSI\CalibrationOutputs\Est_2018-12-17_14-47-14_l1_loss_l2reg_0.001
% 'C:\Users\Amir\Documents\CS SSI\CalibrationOutputs\Est_2018-12-18_14-53-45_regFactors'

outDataOrig = {};
for ii=1:numel(calibOutputDirs)
    tmp = getPythonCalibrationOutput(calibOutputDirs{ii});
    outDataOrig(end+1:end+numel(tmp)) = tmp;
end
origFilters = [];

clear tmp
clear ii
%% load original filters:
origPath = 'C:\Users\Amir\Documents\CS SSI\CalibrationOutputs\OpticalFilters.rawImage';
origFilters = Load3DRawImage(origPath);
origFilters = origFilters(2:end, :);
%% filter data:
outData = outDataOrig;
keepNFilts = [];%[221, 271, 301, 351];
keepweights = {'None'}; %{'quad', 'exp'};
keeplosses = {};%{'l2', 'l1'};
keepregs = {};

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
% subplot(4,1,1);
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
% selected wavelength:
% selected_wavelengths = 1:31;
selected_wavelengths = [6, 15, 22];
plotii=2;

for gamma=selected_wavelengths
    
    figure('units','normalized','outerposition',[0 0 1 1]);
%     subplot(4,1,plotii);
    hold on;
    filtLegends = {};
    for ii=1:numel(outData)
        filter = outData{ii}.Filters(gamma,:);
%         filter = filter / max(filter);
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

%% save calibration matrixes:
outputDir = 'C:\Users\Amir\Documents\CS SSI\ImageDatabase\CalibrationDatabase\l1_loss\\crop25';
NAx = 7936;
NAy = 2592;
for ii=1:numel(outData)
    outfilters = outData{ii}.Filters;
    saveName = fullfile(outputDir, ['A_est_' outData{ii}.name]);
    A_est = PythonCalibrationToBlockToeplitz(outfilters, NAy, NAx);
    save(saveName, 'A_est');
end

%% compare calibration matrices to filtering using the filters:
% get filters and A matrix:
for ii=1:numel(outData)
    NAx = 31*256;
    NAy = 2592;
    f = outData{ii}.Filters;
    A_est = PythonCalibrationToBlockToeplitz(f, NAy, NAx);
    
    % get a 'cube' image:
    cubepath = 'C:\Users\Amir\Documents\CS SSI\ImageDatabase\Test\Image_1_Cube.rawImage';
    cubeim = Load3DRawImage(cubepath);
    
    % cube image -> X image:
    X_im = [];
    for ii=1:size(cubeim,3)
        X_im = [X_im; cubeim(:,:,ii)'];
    end
    
    % compute Y using Blockwise Toeplitz matrix:
    Y_est1 = (A_est * X_im)';
    
    % compute Y using convolution:
    Y_est2 = zeros(size(Y_est1));
    convsize = size(f,2) + size(cubeim,2) - 1;
    ibegin = (size(Y_est2,2) - convsize)/2 +1;
    iend =  ibegin + convsize - 1;
    for ii=1:size(f, 1)
        Y_est2(:,ibegin:iend) = Y_est2(:,ibegin:iend) + ...
            conv2(1, fliplr(f(ii,:)), cubeim(:,:,ii));
    end
    
    % check the difference:
    diff = sum(abs(Y_est2(:) - Y_est1(:)))/ numel(Y_est2);
    disp(['diff is: ' num2str(diff)]);
    %  figure; subplot(2,1,1); imshow(Y_est1, []); subplot(2,1,2); imshow(Y_est2, []);
end
%% crop the end of the filters:
crop_coeffs = 25; % remove crop_coeffs from the beginning and end of filter
inds_bad = [];
for ii=1:numel(outData)
    % make sure there are enough coefficients to remove:
    if (outData{ii}.NFilt <= 2*crop_coeffs)
        inds_bad(end+1) = ii;
        continue;
    end
    % remove them:
    outData{ii}.NFilt = outData{ii}.NFilt - 2*crop_coeffs;
    outData{ii}.Filters = outData{ii}.Filters(:,crop_coeffs+1:end-crop_coeffs);
    outData{ii}.name = [outData{ii}.name '_crop_coeffs_' num2str(crop_coeffs)];
end
