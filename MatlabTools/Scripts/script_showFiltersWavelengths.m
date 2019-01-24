%% get filters
%clear all;clc;
filtpath = 'C:\Users\Amir\Documents\CS SSI\CalibrationOutputs\newfold\2018_11_18\Filters_2018-11-15_17-00-09.rawImage';
calib_filters = Load3DRawImage(filtpath);

%% show normalized filters shape by wavelength
calib_max = max(abs(calib_filters),[], 2);
calib_filters_norm = calib_filters ./ repmat(calib_max, 1, size(calib_filters,2));
%calib_filters_norm = calib_filters;
xsize = size(calib_filters_norm, 2);
xsize = (xsize - 1) / 2;
xx = -xsize:1:xsize;
for ii = 1:6
    figure('units','normalized','outerposition',[0 0 1 1]);
    for jj=1:5
        subplot(5,1,jj);
        plot(xx, calib_filters_norm((ii-1)*5 + jj,:));
        title(['filter for wavelength' num2str(((ii-1)*5 + jj)*10 + 400)]);
    end
end

%% show the energy of each wavelength
wavelengths = 400 + 10*(1:31);
energies = sum(calib_filters, 2);
figure; plot(wavelengths, energies);
title('energy for wavelengths')

%% show correlation matrix:
Corr = corrcoef(calib_filters');
figure;imshow(Corr,[]);
