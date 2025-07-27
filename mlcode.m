% --- SCRIPT: Prepare_and_Analyze_KMeans.m ---
% (You might have named this mlcode.m)
clc; clear; close all;

% =========================================================================
% SECTION 1: USER CONFIGURATION
% =========================================================================
base_data_path = 'C:\Users\Viratpramodh\Documents\MATLAB'; % Ensure this path is correct
condition_folders_info = {
    struct('subfolder', 'healthy',    'label', 0, 'label_name', 'Healthy'),
    struct('subfolder', 'unhealthy1', 'label', 1, 'label_name', 'Unhealthy Type 1'),
    struct('subfolder', 'unhealthy2', 'label', 2, 'label_name', 'Unhealthy Type 2')
};
sensor_data_variable_name = 'log_virtual_tip_disp'; % VERIFY THIS
num_dominant_freqs_to_track_ml = 3; % Used by feature extractor

% K-Means Specific Configuration
num_clusters_k = 3; % Set k to the number of expected conditions

% =========================================================================
% SECTION 2: AUTOMATICALLY DISCOVER DATA FILES
% =========================================================================
data_files_info_list = {};
disp('--- Discovering data files from specified subfolders ---');
for i_cond = 1:length(condition_folders_info)
    current_condition = condition_folders_info{i_cond};
    current_subfolder_path = fullfile(base_data_path, current_condition.subfolder);
    fprintf('Scanning folder: %s for condition: "%s" (Label: %d)\n', current_subfolder_path, current_condition.label_name, current_condition.label);
    if ~exist(current_subfolder_path, 'dir')
        warning('Subfolder "%s" does not exist. Skipping.', current_subfolder_path); continue;
    end
    mat_files_in_subfolder = dir(fullfile(current_subfolder_path, '*.mat'));
    if isempty(mat_files_in_subfolder)
        warning('No .mat files found in "%s".', current_subfolder_path); continue;
    end
    files_found_count = 0;
    for j_file = 1:length(mat_files_in_subfolder)
        if mat_files_in_subfolder(j_file).isdir, continue; end
        new_entry = struct('filepath', fullfile(current_subfolder_path, mat_files_in_subfolder(j_file).name), ...
                           'label', current_condition.label, 'label_name', current_condition.label_name);
        data_files_info_list{end+1} = new_entry;
        files_found_count = files_found_count + 1;
    end
    fprintf('  Found %d .mat files for condition "%s".\n', files_found_count, current_condition.label_name);
end
if isempty(data_files_info_list)
    error('CRITICAL: No .mat files found. Check SECTION 1 configuration and paths.');
end
disp('--- Finished discovering data files ---');
fprintf('Total .mat files to process: %d\n', length(data_files_info_list));

% =========================================================================
% SECTION 3: FEATURE EXTRACTION
% =========================================================================
all_features_list = [];
original_labels = []; 
disp('--- Starting Feature Extraction ---');
for i = 1:length(data_files_info_list)
    current_file_info = data_files_info_list{i};
    fprintf('Processing file: %s (True Label: %s)\n', current_file_info.filepath, current_file_info.label_name);
    try
        load_data = load(current_file_info.filepath);
        if ~isfield(load_data, 'Fs'), warning('Fs not found in %s. Skipping file.', current_file_info.filepath); continue; end
        Fs_ml = load_data.Fs;
        if ~isfield(load_data, sensor_data_variable_name), warning('Variable "%s" not found in %s. Skipping file.', sensor_data_variable_name, current_file_info.filepath); continue; end
        sensor_signal_raw = load_data.(sensor_data_variable_name);
        
        sensor_signal_for_features = [];
        if min(size(sensor_signal_raw)) == 1 || isvector(sensor_signal_raw)
            sensor_signal_for_features = sensor_signal_raw;
        else 
             if (strcmp(sensor_data_variable_name, 'log_full_system_disp') || strcmp(sensor_data_variable_name, 'full_disp_log')) ...
                && isfield(load_data, 'free_dofs') && isfield(load_data,'n') 
                tip_dof_global = 2*(load_data.n+1) - 1; 
                tip_dof_idx_local = find(load_data.free_dofs == tip_dof_global, 1);
                if ~isempty(tip_dof_idx_local) && tip_dof_idx_local <= size(sensor_signal_raw,1)
                    sensor_signal_for_features = sensor_signal_raw(tip_dof_idx_local,:);
                else
                    warning('  Could not ID tip DOF in %s for %s. Using 1st row.', sensor_data_variable_name, current_file_info.filepath);
                    sensor_signal_for_features = sensor_signal_raw(1,:); 
                end
             else
                warning('  Variable %s in %s is matrix. Using 1st row.', sensor_data_variable_name, current_file_info.filepath);
                sensor_signal_for_features = sensor_signal_raw(1,:); 
             end
        end
        if isempty(sensor_signal_for_features), warning('  Signal empty for %s. Skipping file.', current_file_info.filepath); continue; end
        sensor_signal_for_features = sensor_signal_for_features(:)'; 

        features_for_current_run = extract_single_signal_features(sensor_signal_for_features, Fs_ml, num_dominant_freqs_to_track_ml, current_file_info.filepath, 1);
        
        if ~isempty(features_for_current_run) && ~all(isnan(features_for_current_run)) 
            all_features_list = [all_features_list; features_for_current_run];
            original_labels = [original_labels; current_file_info.label]; 
        else
            warning('  No valid features (all NaNs or empty) for %s. Skipping sample from this file.', current_file_info.filepath);
        end
    catch ME
        fprintf('  ERROR processing file ''%s'': %s\n', current_file_info.filepath, ME.message);
        if ~isempty(ME.stack)
            fprintf('  Error occurred in file: %s, function: %s, line: %d\n', ME.stack(1).file, ME.stack(1).name, ME.stack(1).line);
        end
    end
end
disp('--- Feature Extraction Finished ---');

if isempty(all_features_list)
    error('CRITICAL: No features extracted from any file. Cannot perform clustering.');
end
fprintf('Total feature sets extracted: %d\n', size(all_features_list, 1));

% =========================================================================
% SECTION 4: DATA PREPARATION FOR K-MEANS
% =========================================================================
X = all_features_list;
col_means_for_imputation = []; 
X_mean_zscore = []; X_std_zscore = []; % For saving scaling parameters

if any(isnan(X(:)))
    warning('NaN values found in feature data. Imputing with column means for K-Means.');
    col_means_for_imputation = zeros(1, size(X, 2));
    for col = 1:size(X, 2)
        col_data = X(:, col);
        mean_val = mean(col_data(~isnan(col_data)));
        if isnan(mean_val) 
            mean_val = 0; 
            warning('Column %d in features is all NaN. Imputing with 0.', col);
        end
        X(isnan(col_data), col) = mean_val;
        col_means_for_imputation(col) = mean_val; 
    end
end
if any(isinf(X(:)))
    warning('Inf values found in feature data. Replacing with large finite numbers.');
    X(isinf(X) & X > 0) = realmax/10; 
    X(isinf(X) & X < 0) = -realmax/10;
end

% --- Feature Scaling (Z-score normalization) ---
disp('Applying Z-score normalization to features for K-Means.');
X_mean_zscore = mean(X, 1); % Calculate mean for each feature (column)
X_std_zscore = std(X, 0, 1);  % Calculate std for each feature (column), flag 0 for N-1 normalization
X_std_zscore(X_std_zscore < eps) = 1; % Replace zero or very small std with 1 to avoid division by zero/Inf
                                     % A small eps (e.g., 1e-12) is better than direct == 0 for robustness
X_for_kmeans = (X - X_mean_zscore) ./ X_std_zscore;
% Check if scaling introduced NaNs (e.g., if a column was constant and std was 0, then became 1)
if any(isnan(X_for_kmeans(:)))
    warning('NaNs introduced after Z-score scaling (likely due to zero std dev columns). Replacing these NaNs with 0.');
    X_for_kmeans(isnan(X_for_kmeans)) = 0;
end
% --- End of Feature Scaling ---

% =========================================================================
% SECTION 5: PERFORM K-MEANS CLUSTERING
% =========================================================================
disp(' '); disp(['--- Performing K-Means Clustering with k = ', num2str(num_clusters_k), ' ---']);
cluster_indices = []; cluster_centroids = []; % Initialize in case of error

try
    rng('default'); 
    opts = statset('Display','final', 'MaxIter', 200);
    
    [cluster_indices, cluster_centroids, sumd, D] = kmeans(X_for_kmeans, num_clusters_k, ...
                                               'Replicates', 5, ...
                                               'Options', opts, ...
                                               'Distance', 'sqEuclidean');
    disp('K-Means clustering complete.');
catch ME_kmeans
    warning('Error during K-Means clustering: %s\nCheck if X_for_kmeans is empty or has too few rows.', ME_kmeans.message);
    % Allow script to continue to save what it can, but results will be incomplete.
end

% =========================================================================
% SECTION 6: ANALYZE CLUSTER RESULTS
% =========================================================================
disp(' '); disp('--- Analyzing Cluster Results ---');
cluster_meanings = strings(num_clusters_k, 1); % Initialize

if isempty(cluster_indices)
    warning('K-Means did not produce cluster_indices. Skipping cluster analysis.');
else
    disp('Number of data points per cluster:');
    for k_idx = 1:num_clusters_k
        fprintf('  Cluster %d: %d data points\n', k_idx, sum(cluster_indices == k_idx));
    end

    disp(' ');
    disp('Cross-tabulation of True Labels vs. K-Means Cluster Assignments:');
    if length(original_labels) ~= length(cluster_indices)
        warning('Mismatch between number of original labels and cluster assignments. Cannot perform detailed cross-tabulation.');
    else
        try
            contingency_table = crosstab(original_labels, cluster_indices);
            disp('Rows: True Labels (as per condition_folders_info)');
            disp('Cols: K-Means Cluster Index');
            disp(contingency_table);

            disp(' ');
            disp('Attempting to map K-Means clusters to original conditions:');
            for k_idx = 1:num_clusters_k
                points_in_cluster_k_mask = (cluster_indices == k_idx);
                labels_in_cluster_k = original_labels(points_in_cluster_k_mask);
                if isempty(labels_in_cluster_k)
                    cluster_meanings(k_idx) = sprintf("Cluster %d: Empty", k_idx);
                    fprintf('  K-Means Cluster %d: Empty Cluster\n', k_idx);
                    continue;
                end
                
                [majority_label, freq] = mode(labels_in_cluster_k);
                purity = freq / length(labels_in_cluster_k);
                
                label_name_found = 'Unknown Original Label';
                for k_info_map = 1:length(condition_folders_info)
                    if condition_folders_info{k_info_map}.label == majority_label
                        label_name_found = condition_folders_info{k_info_map}.label_name;
                        break;
                    end
                end
                cluster_meanings(k_idx) = sprintf('Cluster %d maps to "%s" (Orig Label %d), Purity: %.2f', k_idx, label_name_found, majority_label, purity);
                fprintf('  %s\n', cluster_meanings(k_idx));
            end
        catch ME_crosstab
            warning('Could not generate detailed cross-tabulation/mapping: %s', ME_crosstab.message);
        end
    end

    % Visualization
    num_features_for_plot = size(X_for_kmeans, 2);
    if num_features_for_plot == 2
        figure;
        gscatter(X_for_kmeans(:,1), X_for_kmeans(:,2), cluster_indices);
        hold on;
        if ~isempty(cluster_centroids)
            plot(cluster_centroids(:,1), cluster_centroids(:,2), 'kx', 'MarkerSize', 15, 'LineWidth', 3);
        end
        title('K-Means Clustering (2 Features)');
        xlabel('Feature 1 (Scaled)'); ylabel('Feature 2 (Scaled)');
        legend_entries_plot = cellstr("Cluster " + string(unique(cluster_indices)))'; 
        if ~isempty(cluster_centroids), legend_entries_plot{end+1} = 'Centroids'; end
        legend(legend_entries_plot, 'Location', 'best');
        grid on;
    elseif num_features_for_plot == 3
        figure;
        unique_clusters_plot = unique(cluster_indices); % Get actual cluster IDs present
        colors_plot = lines(length(unique_clusters_plot)); % Colors for actual number of clusters
        legend_handles_3d = [];
        legend_labels_3d = {};
        for i_plt = 1:length(unique_clusters_plot)
            k_val = unique_clusters_plot(i_plt); % Actual cluster number (e.g., 1, 2, 3)
            h = scatter3(X_for_kmeans(cluster_indices==k_val,1), ...
                         X_for_kmeans(cluster_indices==k_val,2), ...
                         X_for_kmeans(cluster_indices==k_val,3), ...
                         36, colors_plot(i_plt,:), 'filled'); % Use i_plt for color indexing
            legend_handles_3d = [legend_handles_3d, h];
            legend_labels_3d{end+1} = sprintf('Cluster %d', k_val);
            hold on;
        end
        if ~isempty(cluster_centroids)
            plot3(cluster_centroids(:,1), cluster_centroids(:,2), cluster_centroids(:,3), ...
                  'kx', 'MarkerSize', 15, 'LineWidth', 3);
            % Find the handle for the centroid plot to include in legend
            h_centroids = findobj(gca, 'Type', 'line', 'Marker', 'x', 'Color', 'k');
            if ~isempty(h_centroids)
                legend_handles_3d = [legend_handles_3d, h_centroids(1)]; % Take the first one if multiple
                legend_labels_3d{end+1} = 'Centroids';
            end
        end
        title('K-Means Clustering (3 Features)');
        xlabel('Feature 1 (Scaled)'); ylabel('Feature 2 (Scaled)'); zlabel('Feature 3 (Scaled)');
        legend(legend_handles_3d, legend_labels_3d, 'Location', 'best');
        grid on; view(3);
    else % Higher dimensions - t-SNE visualization
        disp(' ');
        disp('Data has >3 features. Consider PCA or t-SNE for visualization.');
        
        % --- ISOLATED LICENSE TEST AND T-SNE BLOCK ---
        disp('DEBUG: About to test license function directly...');
        is_licensed_stats_toolbox = false; % Assume not licensed initially
        try
            toolbox_name_to_check = 'Statistics_and_Machine_Learning_Toolbox';
            % Check for interfering variables (less likely for built-in strings but good practice)
            clear test Statistics_and_Machine_Learning_Toolbox; % Attempt to clear if they exist as vars

            is_licensed_stats_toolbox = license('test', toolbox_name_to_check);
            fprintf('DEBUG: Result of license(''test'', ''%s''): %d (1=yes, 0=no)\n', toolbox_name_to_check, is_licensed_stats_toolbox);
        catch ME_license_direct_test
             fprintf('ERROR directly calling license function: %s\n', ME_license_direct_test.message);
             disp(ME_license_direct_test.getReport);
             is_licensed_stats_toolbox = false; % Treat as not licensed if error
        end

        if is_licensed_stats_toolbox && size(X_for_kmeans,1) > 1 % t-SNE needs N > 1
            disp('DEBUG: Statistics and Machine Learning Toolbox IS available and data size is okay for t-SNE.');
            disp('Attempting t-SNE visualization...');
            Y_tsne = []; % Initialize
            try
                perplexity_val = min(30, floor(size(X_for_kmeans,1)/3.1)-1); 
                if perplexity_val < 1 && size(X_for_kmeans,1) > 3, perplexity_val = 1; 
                elseif perplexity_val < 1 
                    warning('Too few samples (%d) for meaningful t-SNE perplexity. Skipping t-SNE.', size(X_for_kmeans,1));
                end

                if perplexity_val >= 1 && size(X_for_kmeans,1) > 3*perplexity_val && size(X_for_kmeans,2) > 1
                    num_pca_comps = min(50, size(X_for_kmeans,2)-1);
                    if num_pca_comps < 1 && size(X_for_kmeans,2) > 1, num_pca_comps = 1; 
                    elseif num_pca_comps < 1
                         warning('Too few features (%d) for PCA reduction in t-SNE. Skipping t-SNE.', size(X_for_kmeans,2));
                         perplexity_val = 0; % Force skip
                    end

                    if perplexity_val >=1 % Proceed if perplexity is still valid
                        Y_tsne = tsne(X_for_kmeans, 'Algorithm','barneshut', ...
                                        'NumPCAComponents', num_pca_comps, ...
                                        'Perplexity', perplexity_val);
                    end
                elseif isempty(Y_tsne) 
                     warning('Conditions for t-SNE not met (e.g., N <= 3*perplexity or too few features). Skipping t-SNE.');
                end

                 if ~isempty(Y_tsne)
                    figure;
                    unique_cluster_ids_tsne = unique(cluster_indices);
                    num_found_clusters_tsne = length(unique_cluster_ids_tsne);
                    colors_for_plot_tsne = lines(max(num_found_clusters_tsne,1)); % Ensure lines gets at least 1
                    
                    h_scatter_tsne = gscatter(Y_tsne(:,1), Y_tsne(:,2), cluster_indices, colors_for_plot_tsne, '.', 15, 'on');
                    
                    title('t-SNE plot of K-Means Clusters');
                    xlabel('t-SNE Dimension 1'); 
                    ylabel('t-SNE Dimension 2');
                    
                    legend_handles_tsne = h_scatter_tsne;
                    legend_labels_tsne = cell(1, num_found_clusters_tsne);
                    for i_tsne = 1:num_found_clusters_tsne
                        actual_cluster_num = unique_cluster_ids_tsne(i_tsne);
                        legend_labels_tsne{i_tsne} = sprintf('Cluster %d', actual_cluster_num);
                    end
                    if ~isempty(legend_handles_tsne) % Only create legend if handles exist
                        legend(legend_handles_tsne, legend_labels_tsne, 'Location', 'best');
                    end
                    grid on;
                 else
                    disp('t-SNE calculation was skipped or returned empty, no t-SNE plot generated.');
                 end
            catch ME_tsne_block
                warning('Error during t-SNE visualization block: %s', ME_tsne_block.message);
                disp(ME_tsne_block.getReport);
            end
        else
            if ~(size(X_for_kmeans,1) > 1)
                disp('DEBUG: Data size not sufficient for t-SNE.');
            elseif ~is_licensed_stats_toolbox
                disp('DEBUG: Statistics and Machine Learning Toolbox NOT available. Skipping t-SNE.');
            end
        end
        disp('DEBUG: Finished license test and t-SNE attempt section.');
        % --- END OF ISOLATED LICENSE TEST AND T-SNE BLOCK ---
    end
end % End of if isempty(cluster_indices) for skipping analysis

% =========================================================================
% SECTION 7: SAVE K-MEANS RESULTS
% =========================================================================
kmeans_output_filename = 'trained_KMeans_results.mat';
saved_cluster_info = struct();
if ~isempty(cluster_centroids)
    saved_cluster_info.centroids = cluster_centroids;
else
    saved_cluster_info.centroids = []; % Save empty if K-Means failed
end
saved_cluster_info.num_clusters = num_clusters_k;
saved_cluster_info.feature_extraction_settings = struct('num_dominant_freqs', num_dominant_freqs_to_track_ml);
if exist('cluster_meanings', 'var') 
    saved_cluster_info.cluster_meanings_derived = cluster_meanings;
else
    saved_cluster_info.cluster_meanings_derived = strings(num_clusters_k,1); 
end
if ~isempty(col_means_for_imputation)
    saved_cluster_info.feature_imputation_means = col_means_for_imputation;
end
% Save scaling parameters
if ~isempty(X_mean_zscore) && ~isempty(X_std_zscore)
    saved_cluster_info.scaling_zscore_mean = X_mean_zscore;
    saved_cluster_info.scaling_zscore_std = X_std_zscore;
end

save(kmeans_output_filename, 'saved_cluster_info');
disp(' ');
fprintf('K-Means clustering results (centroids, etc.) saved to: %s\n', kmeans_output_filename);
disp('This can be used for assigning new data to clusters.');