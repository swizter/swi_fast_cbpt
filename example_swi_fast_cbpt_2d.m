%% Example: 2-D matrix CBPT with a synthetic irregular region
% This script creates paired 2-D feature data.
% An irregular effect region is injected into group 1.
% The significant mask is plotted as blue points over a gray feature grid.

clear; clc; close all;

addpath(fileparts(mfilename('fullpath')));

%% Create synthetic data
rng(4);

num_sample = 24;
feature_size = [80, 80];
sampledim = 1;

base_data = randn([num_sample, feature_size]);
X1 = base_data + 0.35 * randn([num_sample, feature_size]);
X2 = base_data + 0.35 * randn([num_sample, feature_size]);

%% Define an irregular 2-D effect region
[grid_x, grid_y] = ndgrid(1:feature_size(1), 1:feature_size(2));

main_blob = ((grid_x - 42) ./ 14) .^ 2 + ((grid_y - 42) ./ 10) .^ 2 <= 1;
upper_lobe = ((grid_x - 34) ./ 7) .^ 2 + ((grid_y - 35) ./ 6) .^ 2 <= 1;
right_lobe = ((grid_x - 50) ./ 8) .^ 2 + ((grid_y - 48) ./ 7) .^ 2 <= 1;
tail_lobe = ((grid_x - 45) ./ 5) .^ 2 + ((grid_y - 28) ./ 8) .^ 2 <= 1;

true_mask = main_blob | upper_lobe | right_lobe | tail_lobe;

% Make the outline slightly irregular
true_mask(38:42, 38:41) = false;
true_mask(48:51, 41:44) = false;
true_mask(29:34, 45:49) = true;
true_mask(53:57, 52:55) = true;

%% Add an effect to group 1
effect_size = 0.75;
for isample = 1:num_sample
    sample_buffer = squeeze(X1(isample, :, :));
    sample_buffer(true_mask) = sample_buffer(true_mask) + effect_size;
    X1(isample, :, :) = sample_buffer;
end

%% Run CBPT
cfg = [];
cfg.sampledim = sampledim;
cfg.design = 'paired';
cfg.tail = 0;
cfg.clusteralpha = 0.05;
cfg.alpha = 0.05;
cfg.clusterstatistic = 'maxsum';
cfg.numrandomization = 500;
cfg.connectivity = 'axis';
cfg.seed = 5;
cfg.verbose = true;

stat = swi_fast_cbpt(cfg, X1, X2);

%% Print cluster summary
fprintf('\nSignificant positive clusters:\n');
disp(make_cluster_table(stat.posclusters, cfg.alpha));

fprintf('\nSignificant negative clusters:\n');
disp(make_cluster_table(stat.negclusters, cfg.alpha));

%% Plot significant mask in 2-D
[x, y] = ind2sub(feature_size, 1:prod(feature_size));

sig_idx = find(stat.mask);
non_sig_idx = find(~stat.mask);

figure('Color', 'w', 'Position', [100, 100, 620, 620]);
hold on;

scatter(y(non_sig_idx), x(non_sig_idx), ...
    10, [0.78, 0.78, 0.78], 'filled', ...
    'MarkerFaceAlpha', 0.18, 'MarkerEdgeAlpha', 0.18);

if isempty(sig_idx)
    text(feature_size(2) / 2, feature_size(1) / 2, ...
        'No significant cluster', 'HorizontalAlignment', 'center');
else
    scatter(y(sig_idx), x(sig_idx), ...
        34, [0.00, 0.30, 0.95], 'filled', ...
        'MarkerFaceAlpha', 0.95, 'MarkerEdgeAlpha', 0.95);
end

axis image;
set(gca, 'YDir', 'reverse');
grid on;
xlabel('Feature 2');
ylabel('Feature 1');
title('2-D CBPT Significant Mask');
legend({'Not significant', 'Significant'}, 'Location', 'northeastoutside');
box on;

%% Plot t-value map in 2-D
figure('Color', 'w', 'Position', [760, 100, 620, 620]);

t_value = stat.stat(:);
scatter(y, x, 26, t_value, 'filled');

axis image;
set(gca, 'YDir', 'reverse');
grid on;
xlabel('Feature 2');
ylabel('Feature 1');
title('2-D CBPT T-value Map');
colormap(turbo);
colorbar;
box on;

function cluster_table = make_cluster_table(clusters, alpha)
% Make a compact table for display.
if isempty(clusters)
    cluster_table = table();
    return
end

prob = [clusters.prob]';
clusterstat = [clusters.clusterstat]';
cluster_size = [clusters.size]';
peakstat = [clusters.peakstat]';
peakindex = [clusters.peakindex]';
significant = prob < alpha;

cluster_table = table(prob, clusterstat, cluster_size, peakstat, peakindex, significant);
cluster_table = cluster_table(cluster_table.significant, :);
cluster_table = sortrows(cluster_table, 'prob', 'ascend');
end
