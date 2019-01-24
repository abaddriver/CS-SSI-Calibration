%% folders of diffusers:
clear all; clc;
folders = {'C:\Users\Amir\Documents\CS SSI\ImageDatabase\CalibrationDatabase', ...
    'C:\Users\Amir\Documents\CS SSI\ImageDatabase\CalibrationDatabase\crop25'};
%% get all diffusers filenames and paths:
allDiffs = [];
for ii=1:numel(folders)
    xfold = folders{ii}; % choose folder
    listings = dir(xfold);
    
    filtfunc = @(x) isfile(fullfile(x.folder, x.name));
    listings = listings(arrayfun(filtfunc, listings));
    allDiffs = [allDiffs; listings];
end

%% get an image:
imageName = 'C:\Users\Amir\Documents\CS SSI\ImageDatabase\Train\Image_2';
Cube = Load3DRawImage([imageName '_Cube.rawImage']);
DD = Load3DRawImage([imageName '_DD.rawImage']);

%% choose a diffuser and read it:
all_mse = [];
for ii=1:numel(allDiffs)
%ii=26;
A_est = load(fullfile(allDiffs(ii).folder, allDiffs(ii).name));
A_est = A_est.A_est;

% filter Cube with Filt:
DD_est = FilterCubeWithMatrix(Cube, A_est);
% crop DD and DD_est:
NFilt = 301; NXx = size(Cube,2);
% conv_size = NXx + NFilt - 1;
conv_size = 2592;

crop_start_ind = (size(DD,2) - conv_size)/2 + 1;
crop_end_ind = crop_start_ind + conv_size - 1;
DD_crop = DD(:,crop_start_ind:crop_end_ind);
DD_est_crop = DD_est(:,crop_start_ind:crop_end_ind);
%

all_mse(end+1) = sqrt(sum((DD_crop(:) - DD_est_crop(:)).^2)) / numel(DD_crop);
end
%% add mse results to allDiffs:
for ii=1:numel(allDiffs)
    allDiffs(ii).mse = all_mse(ii);
end
%% compare results:
% figure;
% subplot(1,2,1); imshow(DD_crop,[]); title('orig image');
% subplot(1,2,2); imshow(DD_est_crop, []); title('Est image');
% figure;
% imshow(abs(DD_crop(3:end,:) - DD_est_crop(3:end,:))./DD_crop(3:end,:), []);
% mse = sqrt(sum((DD_crop(:) - DD_est_crop(:)).^2)) / numel(DD_crop)

%% show filters from matrices:
% previous calibration:
clear all;
load 2015_____5____12____15_____2____45_A.mat
AFilts = A(:,128:256:end); AFilts = AFilts';
AFilts = AFilts(1:end-2,:);
% new calibrations:
load allDiffs_2018_12_11.mat
getmse = @(x) x.mse;
[minMse, iMin] = min(arrayfun(getmse, allDiffs));
A_est = load(fullfile(allDiffs(iMin).folder, allDiffs(iMin).name));
A_est = A_est.A_est;
A_estFilts = A_est(:,128:256:end); A_estFilts = A_estFilts';

%% show energy differences:
AFilts_energy = squeeze(sum(AFilts, 2));
A_estFilts_energy = squeeze(sum(A_estFilts, 2));
figure; hold on;
plot((1:numel(AFilts_energy)), AFilts_energy / max(AFilts_energy));
plot(1:numel(A_estFilts_energy), A_estFilts_energy / max(A_estFilts_energy));
%% show selected channels:
NFilt = size(A_estFilts,2);
if NFilt~=size(AFilts,2)
    disp('Error!!!!!');
end

ii = 22;
yy = AFilts(ii,:);
yy_est = A_estFilts(ii,:);

% crop the middle:
middle = NFilt/2;
NFilt_new = 350;
yy = yy(middle-(NFilt_new/2):middle+(NFilt_new/2));
yy_est = yy_est(middle-(NFilt_new/2):middle+(NFilt_new/2));

xx = (1:(NFilt_new+1)) - (NFilt_new/2);

figure; hold on;
plot(xx, yy_est/sum(yy_est));
plot(xx, yy/sum(yy));
title(['channel #' num2str(ii)]);
legend('est filts', 'optical filts');

