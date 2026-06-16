function stat = swi_fast_cbpt(cfg, X1, X2)
% This function:
% Runs a matrix-based cluster permutation test for two groups of n-D data.
% The sample dimension is selected by cfg.sampledim and all other dimensions
% are treated as the feature space where clusters can be formed.
%
% Input data:
% cfg.sampledim - The dimension containing samples/trials/subjects. Default: 1.
% cfg.design - 'independent', 'paired', or 'onesample'. Default: 'independent'.
% cfg.tail - 1, -1, or 0 for right, left, or two-tailed test. Default: 0.
% cfg.clusteralpha - Point-level threshold for cluster formation. Default: 0.05.
% cfg.alpha - Cluster-level alpha. Default: 0.05.
% cfg.clusterstatistic - 'maxsum', 'maxsize', or 'maxabs'. Default: 'maxsum'.
% cfg.numrandomization - Number of random permutations. Default: 1000.
% cfg.adjacency - Optional feature adjacency matrix or edge list.
% cfg.connectivity - 'axis' or 'diagonal' for grid adjacency. Default: 'axis'.
% cfg.seed - Optional random seed. Default: [].
% cfg.verbose - true or false. Default: true.
%
% X1, X2:
% n+1 dimensional numeric arrays. One dimension is the sample dimension, and
% the other dimensions define the feature space. For cfg.design = 'onesample',
% X1 can be a difference array and X2 can be [].
%
% Output data:
% stat.stat - n-D t-statistic map.
% stat.prob - n-D cluster-level p-value map.
% stat.mask - n-D logical mask of significant clusters.
% stat.posclusters / stat.negclusters - Cluster summary structures.
% stat.posclusterslabelmat / stat.negclusterslabelmat - n-D cluster labels.
% stat.posdistribution / stat.negdistribution - Permutation null distributions.
% stat.cfg - The final configuration structure.
%
% Brief steps:
% 1. Move the sample dimension to the first dimension.
% 2. Flatten the feature dimensions into a feature vector.
% 3. Calculate the observed t-statistic map.
% 4. Form clusters on thresholded points using the adjacency graph.
% 5. Permute labels or signs and record the largest cluster statistic.
% 6. Assign cluster-level p-values and reshape results back to n-D space.
%
% Example:
% cfg = [];
% cfg.sampledim = 1;
% cfg.design = 'paired';
% cfg.numrandomization = 1000;
% stat = swi_fast_cbpt(cfg, X_condition_1, X_condition_2);
%
% Swizter/Codex, 16.06.2026

if nargin < 2
    error('At least cfg and X1 are required.');
end

if nargin < 3
    X2 = [];
end

if ~license('test', 'Statistics_Toolbox')
    error('Statistics and Machine Learning Toolbox is required.');
end

cfg = set_default_cfg(cfg);

if ~isnumeric(X1)
    error('X1 must be numeric.');
end

if ~isempty(X2) && ~isnumeric(X2)
    error('X2 must be numeric or empty.');
end

if ~isempty(cfg.seed)
    rng(cfg.seed);
end

% Move sample dimension to the first dimension and flatten features
[X1_2d, feature_shape] = reshape_to_sample_feature(X1, cfg.sampledim);

if strcmpi(cfg.design, 'onesample') && isempty(X2)
    X2_2d = [];
else
    [X2_2d, feature_shape_2] = reshape_to_sample_feature(X2, cfg.sampledim);
    if ~isequal(feature_shape, feature_shape_2)
        error('X1 and X2 must have the same feature dimensions.');
    end
end

num_feature = prod(feature_shape);
adjacency = prepare_adjacency(cfg, feature_shape, num_feature);

if cfg.verbose
    fprintf('swi_fast_cbpt: %d features, %d permutations.\n', num_feature, cfg.numrandomization);
end

% Calculate observed statistics
[observed_t, observed_p] = calculate_t_map(cfg, X1_2d, X2_2d);
[pos_clusters, pos_labelvec] = find_clusters(observed_t, observed_p, adjacency, cfg, 'pos');
[neg_clusters, neg_labelvec] = find_clusters(observed_t, observed_p, adjacency, cfg, 'neg');

pos_distribution = zeros(cfg.numrandomization, 1);
neg_distribution = zeros(cfg.numrandomization, 1);

% Run permutations
for irand = 1:cfg.numrandomization
    [perm_X1, perm_X2] = permute_data(cfg.design, X1_2d, X2_2d);
    [perm_t, perm_p] = calculate_t_map(cfg, perm_X1, perm_X2);

    perm_pos_clusters = find_clusters(perm_t, perm_p, adjacency, cfg, 'pos');
    perm_neg_clusters = find_clusters(perm_t, perm_p, adjacency, cfg, 'neg');

    pos_distribution(irand) = get_max_cluster_stat(perm_pos_clusters);
    neg_distribution(irand) = get_max_cluster_stat(perm_neg_clusters);

    if cfg.verbose && mod(irand, max(1, floor(cfg.numrandomization / 10))) == 0
        fprintf('swi_fast_cbpt: permutation %d/%d finished.\n', irand, cfg.numrandomization);
    end
end

% Assign cluster-level p-values
pos_clusters = assign_cluster_prob(pos_clusters, pos_distribution);
neg_clusters = assign_cluster_prob(neg_clusters, neg_distribution);

[prob_vec, mask_vec] = make_prob_and_mask(num_feature, pos_clusters, neg_clusters, cfg.alpha);

stat = [];
stat.stat = reshape(observed_t, feature_shape);
stat.prob = reshape(prob_vec, feature_shape);
stat.mask = reshape(mask_vec, feature_shape);
stat.posclusters = pos_clusters;
stat.negclusters = neg_clusters;
stat.posclusterslabelmat = reshape(pos_labelvec, feature_shape);
stat.negclusterslabelmat = reshape(neg_labelvec, feature_shape);
stat.posdistribution = pos_distribution;
stat.negdistribution = neg_distribution;
stat.cfg = cfg;
stat.feature_shape = feature_shape;
stat.dimord = make_dimord(numel(feature_shape));

end

function cfg = set_default_cfg(cfg)
if isempty(cfg)
    cfg = [];
end

cfg = set_field_if_missing(cfg, 'sampledim', 1);
cfg = set_field_if_missing(cfg, 'design', 'independent');
cfg = set_field_if_missing(cfg, 'tail', 0);
cfg = set_field_if_missing(cfg, 'clusteralpha', 0.05);
cfg = set_field_if_missing(cfg, 'alpha', 0.05);
cfg = set_field_if_missing(cfg, 'clusterstatistic', 'maxsum');
cfg = set_field_if_missing(cfg, 'numrandomization', 1000);
cfg = set_field_if_missing(cfg, 'adjacency', []);
cfg = set_field_if_missing(cfg, 'connectivity', 'axis');
cfg = set_field_if_missing(cfg, 'seed', []);
cfg = set_field_if_missing(cfg, 'verbose', true);

valid_designs = {'independent', 'paired', 'onesample'};
if ~any(strcmpi(cfg.design, valid_designs))
    error('cfg.design must be independent, paired, or onesample.');
end

valid_clusterstatistics = {'maxsum', 'maxsize', 'maxabs'};
if ~any(strcmpi(cfg.clusterstatistic, valid_clusterstatistics))
    error('cfg.clusterstatistic must be maxsum, maxsize, or maxabs.');
end

if ~ismember(cfg.tail, [-1, 0, 1])
    error('cfg.tail must be -1, 0, or 1.');
end
end

function cfg = set_field_if_missing(cfg, field_name, default_value)
if ~isfield(cfg, field_name) || isempty(cfg.(field_name))
    cfg.(field_name) = default_value;
end
end

function [X_2d, feature_shape] = reshape_to_sample_feature(X, sampledim)
if isempty(X)
    X_2d = [];
    feature_shape = [];
    return
end

if sampledim < 1 || sampledim > ndims(X)
    error('cfg.sampledim is outside the dimensions of the input data.');
end

dim_order = 1:ndims(X);
dim_order = [sampledim, dim_order(dim_order ~= sampledim)];
X_perm = permute(X, dim_order);
data_size = size(X_perm);

num_sample = data_size(1);
feature_shape = data_size(2:end);
X_2d = reshape(X_perm, num_sample, []);
end

function adjacency = prepare_adjacency(cfg, feature_shape, num_feature)
if ~isempty(cfg.adjacency)
    adjacency = cfg.adjacency;

    if isnumeric(adjacency) && size(adjacency, 2) == 2 && size(adjacency, 1) ~= size(adjacency, 2)
        edge_i = adjacency(:, 1);
        edge_j = adjacency(:, 2);
        adjacency = sparse(edge_i, edge_j, true, num_feature, num_feature);
    end

    if ~isequal(size(adjacency), [num_feature, num_feature])
        error('cfg.adjacency must be a feature x feature matrix or an edge list.');
    end

    adjacency = sparse(logical(adjacency));
    adjacency = adjacency | adjacency';
    adjacency = adjacency - diag(diag(adjacency));
else
    adjacency = build_grid_adjacency(feature_shape, cfg.connectivity);
end
end

function adjacency = build_grid_adjacency(feature_shape, connectivity)
num_feature = prod(feature_shape);
index_grid = reshape(1:num_feature, feature_shape);
edge_i = [];
edge_j = [];

if strcmpi(connectivity, 'axis')
    for idim = 1:numel(feature_shape)
        if feature_shape(idim) < 2
            continue
        end

        sub_1 = repmat({':'}, 1, numel(feature_shape));
        sub_2 = repmat({':'}, 1, numel(feature_shape));
        sub_1{idim} = 1:(feature_shape(idim) - 1);
        sub_2{idim} = 2:feature_shape(idim);

        idx_1 = index_grid(sub_1{:});
        idx_2 = index_grid(sub_2{:});
        edge_i = [edge_i; idx_1(:)]; %#ok<AGROW>
        edge_j = [edge_j; idx_2(:)]; %#ok<AGROW>
    end
elseif strcmpi(connectivity, 'diagonal')
    offsets = make_offsets(numel(feature_shape));
    for ioffset = 1:size(offsets, 1)
        offset = offsets(ioffset, :);
        sub_1 = repmat({':'}, 1, numel(feature_shape));
        sub_2 = repmat({':'}, 1, numel(feature_shape));

        for idim = 1:numel(feature_shape)
            if offset(idim) == -1
                sub_1{idim} = 2:feature_shape(idim);
                sub_2{idim} = 1:(feature_shape(idim) - 1);
            elseif offset(idim) == 1
                sub_1{idim} = 1:(feature_shape(idim) - 1);
                sub_2{idim} = 2:feature_shape(idim);
            end
        end

        idx_1 = index_grid(sub_1{:});
        idx_2 = index_grid(sub_2{:});
        edge_i = [edge_i; idx_1(:)]; %#ok<AGROW>
        edge_j = [edge_j; idx_2(:)]; %#ok<AGROW>
    end
else
    error('cfg.connectivity must be axis or diagonal.');
end

adjacency = sparse(edge_i, edge_j, true, num_feature, num_feature);
adjacency = adjacency | adjacency';
end

function offsets = make_offsets(ndim_feature)
values = cell(1, ndim_feature);
for idim = 1:ndim_feature
    values{idim} = -1:1;
end

[grids{1:ndim_feature}] = ndgrid(values{:});
offsets = zeros(numel(grids{1}), ndim_feature);
for idim = 1:ndim_feature
    offsets(:, idim) = grids{idim}(:);
end

offsets(all(offsets == 0, 2), :) = [];
offsets = offsets(sum(abs(offsets), 2) > 0, :);
end

function [t_map, p_map] = calculate_t_map(cfg, X1, X2)
switch lower(cfg.design)
    case 'independent'
        n1 = size(X1, 1);
        n2 = size(X2, 1);
        m1 = mean(X1, 1, 'omitnan');
        m2 = mean(X2, 1, 'omitnan');
        v1 = var(X1, 0, 1, 'omitnan');
        v2 = var(X2, 0, 1, 'omitnan');

        se = sqrt(v1 ./ n1 + v2 ./ n2);
        t_map = (m1 - m2) ./ se;
        df = (v1 ./ n1 + v2 ./ n2) .^ 2 ./ ...
            ((v1 ./ n1) .^ 2 ./ (n1 - 1) + (v2 ./ n2) .^ 2 ./ (n2 - 1));

    case 'paired'
        if size(X1, 1) ~= size(X2, 1)
            error('Paired design requires the same number of samples in X1 and X2.');
        end
        D = X1 - X2;
        n = size(D, 1);
        m = mean(D, 1, 'omitnan');
        s = std(D, 0, 1, 'omitnan');
        t_map = m ./ (s ./ sqrt(n));
        df = n - 1;

    case 'onesample'
        if isempty(X2)
            D = X1;
        else
            if size(X1, 1) ~= size(X2, 1)
                error('Onesample design with X2 requires paired X1 and X2.');
            end
            D = X1 - X2;
        end
        n = size(D, 1);
        m = mean(D, 1, 'omitnan');
        s = std(D, 0, 1, 'omitnan');
        t_map = m ./ (s ./ sqrt(n));
        df = n - 1;
end

t_map(~isfinite(t_map)) = 0;

if cfg.tail == 1
    p_map = 1 - tcdf(t_map, df);
elseif cfg.tail == -1
    p_map = tcdf(t_map, df);
else
    p_map = 2 .* (1 - tcdf(abs(t_map), df));
end

p_map(~isfinite(p_map)) = 1;
end

function [clusters, labelvec] = find_clusters(t_map, p_map, adjacency, cfg, polarity)
num_feature = numel(t_map);
labelvec = zeros(1, num_feature);
clusters = struct('prob', {}, 'clusterstat', {}, 'size', {}, 'peakstat', {}, 'peakindex', {}, 'indices', {});

if strcmpi(polarity, 'pos')
    if cfg.tail == -1
        return
    end
    candidate = t_map > 0 & p_map <= cfg.clusteralpha;
else
    if cfg.tail == 1
        return
    end
    candidate = t_map < 0 & p_map <= cfg.clusteralpha;
end

candidate_idx = find(candidate);
if isempty(candidate_idx)
    return
end

sub_adjacency = adjacency(candidate_idx, candidate_idx);
cluster_graph = graph(sub_adjacency, 'upper');
component_id = conncomp(cluster_graph);
num_cluster = max(component_id);

for icluster = 1:num_cluster
    local_idx = find(component_id == icluster);
    feature_idx = candidate_idx(local_idx);
    cluster_t = t_map(feature_idx);

    labelvec(feature_idx) = icluster;
    clusters(icluster).prob = NaN;
    clusters(icluster).clusterstat = calculate_cluster_stat(cluster_t, cfg.clusterstatistic);
    clusters(icluster).size = numel(feature_idx);

    [~, peak_local] = max(abs(cluster_t));
    clusters(icluster).peakstat = cluster_t(peak_local);
    clusters(icluster).peakindex = feature_idx(peak_local);
    clusters(icluster).indices = feature_idx;
end
end

function cluster_stat = calculate_cluster_stat(cluster_t, clusterstatistic)
switch lower(clusterstatistic)
    case 'maxsum'
        cluster_stat = sum(abs(cluster_t));
    case 'maxsize'
        cluster_stat = numel(cluster_t);
    case 'maxabs'
        cluster_stat = max(abs(cluster_t));
end
end

function max_stat = get_max_cluster_stat(clusters)
if isempty(clusters)
    max_stat = 0;
else
    max_stat = max([clusters.clusterstat]);
end
end

function [perm_X1, perm_X2] = permute_data(design, X1, X2)
switch lower(design)
    case 'independent'
        n1 = size(X1, 1);
        X_all = [X1; X2];
        order = randperm(size(X_all, 1));
        perm_X1 = X_all(order(1:n1), :);
        perm_X2 = X_all(order(n1 + 1:end), :);

    case 'paired'
        swap_flag = rand(size(X1, 1), 1) > 0.5;
        perm_X1 = X1;
        perm_X2 = X2;
        perm_X1(swap_flag, :) = X2(swap_flag, :);
        perm_X2(swap_flag, :) = X1(swap_flag, :);

    case 'onesample'
        if isempty(X2)
            D = X1;
        else
            D = X1 - X2;
        end
        sign_flip = (rand(size(D, 1), 1) > 0.5) * 2 - 1;
        perm_X1 = D .* sign_flip;
        perm_X2 = [];
end
end

function clusters = assign_cluster_prob(clusters, distribution)
for icluster = 1:numel(clusters)
    clusters(icluster).prob = (1 + sum(distribution >= clusters(icluster).clusterstat)) ./ ...
        (numel(distribution) + 1);
end
end

function [prob_vec, mask_vec] = make_prob_and_mask(num_feature, pos_clusters, neg_clusters, alpha)
prob_vec = ones(1, num_feature);
mask_vec = false(1, num_feature);

for icluster = 1:numel(pos_clusters)
    idx = pos_clusters(icluster).indices;
    prob_vec(idx) = pos_clusters(icluster).prob;
    if pos_clusters(icluster).prob < alpha
        mask_vec(idx) = true;
    end
end

for icluster = 1:numel(neg_clusters)
    idx = neg_clusters(icluster).indices;
    prob_vec(idx) = neg_clusters(icluster).prob;
    if neg_clusters(icluster).prob < alpha
        mask_vec(idx) = true;
    end
end
end

function dimord = make_dimord(num_dim)
dimord = 'feature1';
for idim = 2:num_dim
    dimord = [dimord, '_feature', num2str(idim)]; %#ok<AGROW>
end
end
