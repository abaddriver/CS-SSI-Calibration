%% save the input filters as matrix:
clear all; clc;
load 2015_____5____12____15_____2____45_A.mat
%% get the filters:
Filters = A( :, 128:256:end)';
NFiltMax = size(Filters,2);
% xx = (1:NFiltMax) - (NFiltMax/2);
% plot(xx, Filters(1:5,:));

%% how many coefficients do we need based on this matrix?
sumFilters = sum(Filters(:));
ratios = zeros(NFiltMax/2 ,1);
NFilts = 2:2:NFiltMax;
for NFilt=NFilts
    idx_begin = (NFiltMax - NFilt)/2 + 1;
    idx_end = idx_begin + NFilt - 1;
    sum_NFilt = sum(sum(Filters(:,idx_begin:idx_end)));
    ratios(NFilt/2) = sum_NFilt / sumFilters;
end

%% are the filters centered? calculate the center of mass:
Filters1 = Filters ./ repmat(sum(Filters,2), [1 size(Filters,2)]);
weights = repmat((1:NFiltMax), [size(Filters1,1) 1]);
means = sum(weights .* Filters1, 2);
mean(means - (NFiltMax/2))
% this results in only ~4 pixels
% not very significant

%% crop the filters for 350 values:
NFilt = 351;
