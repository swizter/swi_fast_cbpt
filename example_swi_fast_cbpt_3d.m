%% Example: 3-D matrix CBPT with synthetic data
% This script creates two groups of samples.
% Each sample contains a 3-D feature matrix: x * y * z.
% A cartoon-like effect is injected into group 1, then swi_fast_cbpt is used to
% detect the significant cluster.

clear; clc; close all;

addpath(fileparts(mfilename('fullpath')));

%% Create synthetic data
rng(1);

num_sample = 24;
feature_size = [24, 30, 16];
sampledim = 1;

base_data = randn([num_sample, feature_size]);
X1 = base_data + 0.35 * randn([num_sample, feature_size]);
X2 = base_data + 0.35 * randn([num_sample, feature_size]);

% Define a cartoon-like 3-D effect region
true_mask = false(feature_size);
[grid_x, grid_y, grid_z] = ndgrid(1:feature_size(1), 1:feature_size(2), 1:feature_size(3));

left_ball = ((grid_x - 8.8) ./ 3.8) .^ 2 + ((grid_y - 7) ./ 3.7) .^ 2 + ((grid_z - 8) ./ 2.8) .^ 2 <= 1;
right_ball = ((grid_x - 15.2) ./ 3.8) .^ 2 + ((grid_y - 7) ./ 3.7) .^ 2 + ((grid_z - 8) ./ 2.8) .^ 2 <= 1;
shaft = grid_x >= 10 & grid_x <= 14 & grid_y >= 7 & grid_y <= 24 & grid_z >= 6 & grid_z <= 10;
tip = ((grid_x - 12) ./ 3.1) .^ 2 + ((grid_y - 24.8) ./ 3.0) .^ 2 + ((grid_z - 8) ./ 2.5) .^ 2 <= 1;

true_mask = left_ball | right_ball | shaft | tip;

% Add a tiny amount of boundary irregularity without changing the main ratio
true_mask(9:10, 10:12, 9:10) = true;
true_mask(14:15, 10:12, 6:7) = true;

% Add an effect to group 1
effect_size = 0.75;
for isample = 1:num_sample
    sample_buffer = squeeze(X1(isample, :, :, :));
    sample_buffer(true_mask) = sample_buffer(true_mask) + effect_size;
    X1(isample, :, :, :) = sample_buffer;
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
cfg.seed = 2;
cfg.verbose = true;

stat = swi_fast_cbpt(cfg, X1, X2);

%% Print cluster summary
fprintf('\nSignificant positive clusters:\n');
disp(make_cluster_table(stat.posclusters, cfg.alpha));

fprintf('\nSignificant negative clusters:\n');
disp(make_cluster_table(stat.negclusters, cfg.alpha));

%% Plot significant mask in 3-D
[x, y, z] = ind2sub(feature_size, 1:prod(feature_size));

sig_idx = find(stat.mask);
non_sig_idx = find(~stat.mask);

figure('Color', 'w', 'Position', [100, 100, 760, 620]);
hold on;

scatter3(x(non_sig_idx), y(non_sig_idx), z(non_sig_idx), ...
    10, [0.78, 0.78, 0.78], 'filled', ...
    'MarkerFaceAlpha', 0.12, 'MarkerEdgeAlpha', 0.12);

if isempty(sig_idx)
    text(0.5, 0.5, 0.5, 'No significant cluster', 'HorizontalAlignment', 'center');
else
    scatter3(x(sig_idx), y(sig_idx), z(sig_idx), ...
        42, [0.00, 0.30, 0.95], 'filled', ...
        'MarkerFaceAlpha', 0.95, 'MarkerEdgeAlpha', 0.95);
end

axis equal;
grid on;
xlabel('Feature 1');
ylabel('Feature 2');
zlabel('Feature 3');
title('CBPT Significant Mask');
legend({'Not significant', 'Significant'}, 'Location', 'northeastoutside');
view(35, 25);
box on;

%% Plot t-value map in 3-D
figure('Color', 'w', 'Position', [880, 100, 760, 620]);

t_value = stat.stat(:);
scatter3(x, y, z, 26, t_value, 'filled', ...
    'MarkerFaceAlpha', 0.85, 'MarkerEdgeAlpha', 0.85);

axis equal;
grid on;
xlabel('Feature 1');
ylabel('Feature 2');
zlabel('Feature 3');
title('3-D CBPT T-value Map');
colormap(turbo);
colorbar;
view(35, 25);
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
