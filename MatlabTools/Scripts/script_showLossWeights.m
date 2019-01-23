%% definitions:
clear all;
NXx = 256;
NFx = 351;
NYx = 606;
conv_size = NXx+NFx-1;
minConvCoeffs = (conv_size - NYx)/2;
maxCoeffs = min(NXx, NFx);
peakLength = conv_size - 2*(maxCoeffs-1);

%% create proportional weights:
proportional_weights = (minConvCoeffs+1):(maxCoeffs-1);
proportional_weights = [proportional_weights maxCoeffs*ones(1,peakLength)];
proportional_weights = [proportional_weights, (maxCoeffs-1):-1:(minConvCoeffs+1)];
proportional_weights = proportional_weights / maxCoeffs;

%% create other weights from it:
none_weights = ones(1, size(proportional_weights,2));
square_weights = proportional_weights.^2;
quad_weights = proportional_weights.^4;
exp_weights = exp((proportional_weights-1.0)*maxCoeffs/20);

%% plot weights:
xx = 1:numel(proportional_weights);

figure; hold on;
plot(xx, none_weights, 'linewidth',4);
plot(xx, proportional_weights, 'linewidth',4);
plot(xx, square_weights, 'linewidth',4);
plot(xx, quad_weights, 'linewidth',4);
plot(xx, exp_weights, 'linewidth',4);

ax = xlabel('horizontal image indices');
ax.FontSize = 40;
ax = ylabel('Loss Weight');
ax.FontSize = 40;

lgd = legend({'none', 'proportional', 'square', 'quad', 'exp'}, ...
    'FontSize', 40);

