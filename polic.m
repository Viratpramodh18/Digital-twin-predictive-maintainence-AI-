% --- MASTER SCRIPT-LIKE POLIC.M: Simulate, Capture, then K-Means Predict ---
clc; clear; close all; % Clear workspace, close figures

% =========================================================================
% PART 0: SHARED CONFIGURATION
% =========================================================================
% --- DAQ & Simulation Config (from your Part 1) ---
simulation_duration_target = 10; % seconds - HOW LONG TO RUN THE SIMULATION FOR CAPTURE
output_signal_to_save_and_analyze = 'log_virtual_tip_disp'; % The variable name to save from sim and analyze later
daq_vendor = 'ni';
daq_device_name_sim = "Dev1"; % Device name for the simulation's DAQ input
daq_channel_id_sim = "ai0";   % Channel for the simulation's DAQ input
accelerometer_sensitivity_sim = 100; % For the simulation's DAQ input channel
Fs_requested_sim = 1652; % Requested Fs for simulation DAQ

% --- K-Means Model & Feature Config (from your Part 2 / polic.m) ---
kmeans_results_filename = 'trained_KMeans_results.mat'; 
% num_dominant_freqs_to_use will be loaded from kmeans_results_filename

% --- For interpreting K-Means cluster meanings ---
label_map_kmeans = { 
    struct('numeric', 0, 'name', 'HEALTHY'), % Assuming original labels were 0,1,2
    struct('numeric', 1, 'name', 'UNHEALTHY - Fault Type 1'),
    struct('numeric', 2, 'name', 'UNHEALTHY - Fault Type 2')
};

% --- Beam properties (for simulation - Part 1) ---
L = 0.3034; E = 69e9; rho = 2700; b = 0.0419; h = 0.0012;
A = b * h; I = (b * h^3) / 12;
n_sim = 99; % Number of elements for simulation
le_sim = L / n_sim;


% =========================================================================
% PART 1: REAL-TIME SIMULATION AND DATA CAPTURE
% =========================================================================
disp('--- PART 1: Starting Beam Simulation and Data Capture ---');

% --- Mass and Stiffness Matrices (for simulation) ---
M_global_sim = zeros(2*(n_sim+1));
K_global_sim = zeros(2*(n_sim+1));
for i_elem = 1:n_sim
    Me_sim = rho*A*le_sim/420 * [156 22*le_sim 54 -13*le_sim; 22*le_sim 4*le_sim^2 13*le_sim -3*le_sim^2;
                         54 13*le_sim 156 -22*le_sim; -13*le_sim -3*le_sim^2 -22*le_sim 4*le_sim^2];
    Ke_sim = E*I/le_sim^3 * [12 6*le_sim -12 6*le_sim; 6*le_sim 4*le_sim^2 -6*le_sim 2*le_sim^2;
                     -12 -6*le_sim 12 -6*le_sim; 6*le_sim 2*le_sim^2 -6*le_sim 4*le_sim^2];
    idx_sim = 2*i_elem-1:2*i_elem+2;
    M_global_sim(idx_sim, idx_sim) = M_global_sim(idx_sim, idx_sim) + Me_sim;
    K_global_sim(idx_sim, idx_sim) = K_global_sim(idx_sim, idx_sim) + Ke_sim;
end
alpha_damp_sim = 0.01; beta_damp_sim = 1e-5; % Damping for simulation
C_global_sim = alpha_damp_sim*M_global_sim + beta_damp_sim*K_global_sim;
free_dofs_sim = 3:2*(n_sim+1);
M_sim = M_global_sim(free_dofs_sim, free_dofs_sim);
K_sim = K_global_sim(free_dofs_sim, free_dofs_sim);
C_sim = C_global_sim(free_dofs_sim, free_dofs_sim);
beta_n_sim = 0.25; gamma_n_sim = 0.5; % Newmark-beta parameters

% --- DAQ Setup for Simulation Input ---
daq_sim = []; Fs_sim_actual = 0;
daq_sim_initialized_successfully = false;
try
    disp('Initializing DAQ for simulation input...');
    daq_sim = daq(daq_vendor); % Using vendor from Part 0
    ch_sim = addinput(daq_sim, daq_device_name_sim, daq_channel_id_sim, "Accelerometer");
    ch_sim.Sensitivity = accelerometer_sensitivity_sim;
    daq_sim.Rate = Fs_requested_sim;
    Fs_sim_actual = daq_sim.Rate;
    if Fs_sim_actual == 0, error('Simulation DAQ rate is zero.'); end
    disp(['Actual Simulation DAQ rate set to: ', num2str(Fs_sim_actual), ' Hz']);
    % cleanupObj_sim = onCleanup(@() stopAndReleaseDaq_local_polic(daq_sim)); % Local function for cleanup
    daq_sim_initialized_successfully = true;
catch ME_daq_sim
    disp('CRITICAL ERROR DURING SIMULATION DAQ INITIALIZATION:'); disp(ME_daq_sim.message);
    if ~isempty(daq_sim) && isvalid(daq_sim), delete(daq_sim); end
    disp('Cannot proceed with simulation. Exiting.');
    return;
end

% --- Calculations dependent on Fs_sim_actual ---
block_time_sim = 1; % Process 1 second of DAQ data at a time for simulation
block_samples_sim = round(Fs_sim_actual * block_time_sim);
if block_samples_sim == 0, error('block_samples_sim is zero. Check Fs_sim_actual.'); end
dt_sim = 1 / Fs_sim_actual;

% --- Plot setup for simulation (optional) ---
enable_live_plots_sim = true; 
figure_handle_sim = [];
if enable_live_plots_sim
    figure_handle_sim = figure('Name','Real-Time Simulation & DAQ Capture','Position',[50 50 1000 600]);
    ax_sim_input = subplot(2,1,1); h_input_accel_sim_daq = plot(ax_sim_input, NaN, NaN, '-b'); title(ax_sim_input, 'Input DAQ Accel (for Sim)'); ylabel(ax_sim_input, 'm/s^2'); grid on;
    ax_sim_output = subplot(2,1,2); h_virtual_tip_disp_sim = plot(ax_sim_output, NaN, NaN, '-r'); title(ax_sim_output, 'Simulated Virtual Tip Disp'); ylabel(ax_sim_output, 'm'); xlabel(ax_sim_output, 'Time (s)'); grid on;
    
    % Initialize plot data buffers if plotting
    N_window_samples_sim = round(max(5, simulation_duration_target + 1) * Fs_sim_actual);
    t_plot_window_sim = (0:N_window_samples_sim-1) * dt_sim;
    input_buffer_sim = zeros(1, N_window_samples_sim);
    disp_buffer_sim = zeros(1, N_window_samples_sim);
    set(h_input_accel_sim_daq, 'XData', t_plot_window_sim, 'YData', input_buffer_sim);
    set(h_virtual_tip_disp_sim, 'XData', t_plot_window_sim, 'YData', disp_buffer_sim);
end

% --- Initial conditions & Logging for simulation ---
u_all_dofs_sim = zeros(length(free_dofs_sim), 1);
v_all_dofs_sim = zeros(length(free_dofs_sim), 1);
% Force applied at the tip DOF for simulation, based on DAQ input
a_all_dofs_sim = M_sim \ (-C_sim*v_all_dofs_sim - K_sim*u_all_dofs_sim); % Initial acceleration assuming zero external force
u_history_for_next_block_sim = u_all_dofs_sim;

% Initialize loggers
sim_log_time = [];
sim_log_input_daq_accel = []; % The DAQ signal used as input force
sim_log_virtual_tip_disp = []; % The simulated output displacement we want to analyze

time_elapsed_total_sim = 0;
disp(['Starting simulation data capture for ~', num2str(simulation_duration_target), ' seconds...']);
start(daq_sim, "continuous"); % Start DAQ for continuous acquisition
pause(0.5); % Allow DAQ to buffer some initial data

captured_data_filename_sim = ''; % To store the name of the saved .mat file from simulation

try
    while time_elapsed_total_sim < simulation_duration_target
        if enable_live_plots_sim && ~ishandle(figure_handle_sim)
            disp('Simulation plot figure closed. Stopping data capture.');
            break;
        end

        sensor_data_matrix_sim = read(daq_sim, block_samples_sim, "OutputFormat", "Matrix");
        block_daq_input_accel_sim = zeros(1, block_samples_sim); % Initialize block input
        if ~isempty(sensor_data_matrix_sim)
            num_samples_read_sim = size(sensor_data_matrix_sim, 1);
            if num_samples_read_sim > 0
                raw_daq_data_vector_sim = sensor_data_matrix_sim(:,1)'; % Assuming single channel accelerometer
                block_daq_input_accel_sim(1:min(num_samples_read_sim, block_samples_sim)) = raw_daq_data_vector_sim(1:min(num_samples_read_sim, block_samples_sim));
                if num_samples_read_sim < block_samples_sim, warning('SIM_DAQ: Read fewer samples (%d) than requested (%d). Padded with zeros.', num_samples_read_sim, block_samples_sim); end
            else, warning('SIM_DAQ: Read 0 samples in this block.'); end
        else, warning('SIM_DAQ: Read returned empty matrix this block.'); end

        % Define where the DAQ input force is applied in the simulation
        % For simplicity, let's assume it excites the tip DOF displacement (as a force proportional to acceleration)
        tip_dof_global_sim = 2*(n_sim+1) - 1; % Tip displacement DOF (vertical)
        tip_dof_idx_local_sim = find(free_dofs_sim == tip_dof_global_sim, 1);
        
        block_force_on_DOFs_sim = zeros(length(free_dofs_sim), block_samples_sim);
        if ~isempty(tip_dof_idx_local_sim)
             % Scale factor for force, assuming DAQ accel is proportional to force.
             % This is a placeholder; you might need a proper transfer function or scaling.
             force_scale_factor = 1.0; % Adjust as needed
             block_force_on_DOFs_sim(tip_dof_idx_local_sim, :) = block_daq_input_accel_sim * force_scale_factor;
        end

        % Newmark-beta integration for this block
        block_system_disp_sim_current = zeros(length(free_dofs_sim), block_samples_sim);
        current_u_sim = u_history_for_next_block_sim; % Use state from end of last block
        current_v_sim = v_all_dofs_sim; 
        current_a_sim = a_all_dofs_sim;

        for k_step = 1:block_samples_sim
            % Standard Newmark-Beta steps
            u_pred_sim = current_u_sim + dt_sim*current_v_sim + (0.5 - beta_n_sim)*dt_sim^2*current_a_sim;
            v_pred_sim = current_v_sim + (1 - gamma_n_sim)*dt_sim*current_a_sim;
            
            LHS_sim = M_sim + gamma_n_sim*dt_sim*C_sim + beta_n_sim*dt_sim^2*K_sim;
            RHS_sim = block_force_on_DOFs_sim(:, k_step) - C_sim*v_pred_sim - K_sim*u_pred_sim;
            
            a_next_sim = LHS_sim \ RHS_sim;
            u_next_sim = u_pred_sim + beta_n_sim*dt_sim^2*a_next_sim;
            v_next_sim = v_pred_sim + gamma_n_sim*dt_sim*a_next_sim;
            
            block_system_disp_sim_current(:, k_step) = u_next_sim;
            current_u_sim = u_next_sim; % Update for next step within block
            current_v_sim = v_next_sim;
            current_a_sim = a_next_sim;
        end
        % Store states for the start of the *next* block
        u_history_for_next_block_sim = block_system_disp_sim_current(:, end);
        v_all_dofs_sim = current_v_sim; % v at end of block
        a_all_dofs_sim = current_a_sim; % a at end of block
        
        % Extract the desired output signal for logging (e.g., virtual tip displacement)
        block_virtual_tip_disp_sim = zeros(1, block_samples_sim);
        if ~isempty(tip_dof_idx_local_sim)
            block_virtual_tip_disp_sim = block_system_disp_sim_current(tip_dof_idx_local_sim, :);
        end

        % Update plot buffers (if enabled)
        if enable_live_plots_sim
            input_buffer_sim = [input_buffer_sim(block_samples_sim+1:end), block_daq_input_accel_sim];
            disp_buffer_sim  = [disp_buffer_sim(block_samples_sim+1:end), block_virtual_tip_disp_sim];
            set(h_input_accel_sim_daq, 'YData', input_buffer_sim); 
            if any(input_buffer_sim), ylim(ax_sim_input, [min(input_buffer_sim)*1.1-eps, max(input_buffer_sim)*1.1+eps]); end
            set(h_virtual_tip_disp_sim, 'YData', disp_buffer_sim); 
            if any(disp_buffer_sim), ylim(ax_sim_output, [min(disp_buffer_sim)*1.1-eps, max(disp_buffer_sim)*1.1+eps]); end
            drawnow limitrate;
        end
        
        % Log data for saving
        current_block_timestamps_sim = time_elapsed_total_sim + (0:block_samples_sim-1)*dt_sim;
        sim_log_time = [sim_log_time, current_block_timestamps_sim];
        time_elapsed_total_sim = current_block_timestamps_sim(end) + dt_sim; % Update total time
        
        sim_log_input_daq_accel = [sim_log_input_daq_accel, block_daq_input_accel_sim];
        % Save the specific output signal needed for analysis
        if strcmp(output_signal_to_save_and_analyze, 'log_virtual_tip_disp')
            sim_log_virtual_tip_disp = [sim_log_virtual_tip_disp, block_virtual_tip_disp_sim];
        % Add other 'elseif' if you might save different signals under output_signal_to_save_and_analyze
        else
            error('Unsupported output_signal_to_save_and_analyze specified for logging.');
        end
        
        fprintf('Simulation time: %.2f / %.2f s\n', time_elapsed_total_sim, simulation_duration_target);

    end % End of while loop for simulation
catch ME_main_sim_loop
    disp("ERROR DURING REAL-TIME SIMULATION LOOP:"); disp(ME_main_sim_loop.message);
    disp(ME_main_sim_loop.getReport);
end

% --- Stop and Save Simulation Data ---
if ~isempty(daq_sim) && isvalid(daq_sim)
    if isprop(daq_sim, 'Running') && daq_sim.Running
        disp('Stopping simulation DAQ...');
        stop(daq_sim);
    end
    delete(daq_sim); % Release DAQ
    clear daq_sim;
end
if enable_live_plots_sim && ishandle(figure_handle_sim)
    % close(figure_handle_sim); % Optionally close figure
end
disp('Simulation data capture finished.');

timestamp_str_sim = datestr(now, 'yyyymmdd_HHMMSS');
captured_data_filename_sim = ['simulated_capture_', timestamp_str_sim, '.mat'];
save_successful_sim = false;
try
    vars_to_save = {'Fs_sim_actual', 'sim_log_time'};
    % Dynamically add the chosen output signal to save list
    vars_to_save{end+1} = output_signal_to_save_and_analyze; % This will save 'log_virtual_tip_disp'
    
    % Create the variable in the current workspace to match its name for saving
    eval([output_signal_to_save_and_analyze, ' = sim_log_virtual_tip_disp;']); % Assuming this is the one
    
    % Also save Fs with the standard name 'Fs' for compatibility with feature extractor
    Fs = Fs_sim_actual; % Create Fs variable
    vars_to_save{end+1} = 'Fs';

    disp(['Saving: ', strjoin(vars_to_save,', ')]);
    save(captured_data_filename_sim, vars_to_save{:}); % Save only specified vars
    disp(['✅ Simulated data saved to ', captured_data_filename_sim]);
    save_successful_sim = true;
catch ME_save_sim
    disp('ERROR DURING SIMULATED DATA SAVING:'); disp(ME_save_sim.message);
    disp(['Attempted to save to: ', captured_data_filename_sim]);
end

if ~save_successful_sim || isempty(captured_data_filename_sim)
    disp('Simulated data saving failed or filename is empty. Cannot proceed with K-Means analysis. Exiting.');
    return;
end
disp('--- PART 1: Finished ---');
fprintf('\n\n');


% =========================================================================
% PART 2: K-MEANS PREDICTION ON CAPTURED DATA (largely from previous polic.m)
% =========================================================================
disp('--- PART 2: Starting K-Means Prediction on Captured Data ---');

% --- Load K-Means Model (Section 2 from previous polic.m) ---
disp(['Loading K-Means results from: ', kmeans_results_filename]);
if ~exist(kmeans_results_filename, 'file')
    error('K-Means results file "%s" not found. Ensure K-Means training script has run.', kmeans_results_filename);
end
try
    loaded_kmeans_data = load(kmeans_results_filename); 
    if isfield(loaded_kmeans_data, 'saved_cluster_info')
        kmeans_info = loaded_kmeans_data.saved_cluster_info;
        disp('K-Means results loaded successfully.');
        cluster_centroids = kmeans_info.centroids;
        num_clusters = kmeans_info.num_clusters;
        cluster_meanings_derived = kmeans_info.cluster_meanings_derived;
        if isfield(kmeans_info, 'feature_extraction_settings') && isfield(kmeans_info.feature_extraction_settings, 'num_dominant_freqs')
            num_dominant_freqs_to_use = kmeans_info.feature_extraction_settings.num_dominant_freqs;
            fprintf('  Using num_dominant_freqs = %d (from saved K-Means settings).\n', num_dominant_freqs_to_use);
        else
            warning('num_dominant_freqs not found in K-Means settings. Using default of 3.');
            num_dominant_freqs_to_use = 3; 
        end
        feature_imputation_means = [];
        if isfield(kmeans_info, 'feature_imputation_means') && ~isempty(kmeans_info.feature_imputation_means)
            feature_imputation_means = kmeans_info.feature_imputation_means;
            disp('  Loaded feature imputation means.');
        end
        scaling_zscore_mean = []; scaling_zscore_std = [];
        if isfield(kmeans_info, 'scaling_zscore_mean') && isfield(kmeans_info, 'scaling_zscore_std') ...
                && ~isempty(kmeans_info.scaling_zscore_mean) && ~isempty(kmeans_info.scaling_zscore_std)
            scaling_zscore_mean = kmeans_info.scaling_zscore_mean;
            scaling_zscore_std = kmeans_info.scaling_zscore_std;
            disp('  Loaded Z-score scaling parameters.');
        else
            disp('  No Z-score scaling parameters found/loaded. Features will not be scaled by this script.');
        end
    else
        error('Variable "saved_cluster_info" not found in K-Means results file "%s".', kmeans_results_filename);
    end
catch ME_load_kmeans_part2
    error('Error loading K-Means results for Part 2: %s', ME_load_kmeans_part2.message);
end

% --- Data to Analyze: The file just saved from Part 1 ---
data_filepath_for_prediction = captured_data_filename_sim;
current_data_source_name_pred = data_filepath_for_prediction;

disp(['--- Analyzing captured file: ', data_filepath_for_prediction, ' ---']);
raw_signal_pred = []; 
signal_Fs_for_extraction_pred = NaN;

try
    loaded_capture_data = load(data_filepath_for_prediction);
    % The variable name is output_signal_to_save_and_analyze (e.g., 'log_virtual_tip_disp')
    % And Fs was saved as 'Fs'
    if ~isfield(loaded_capture_data, output_signal_to_save_and_analyze)
        error('Signal variable "%s" not found in captured data file "%s".', output_signal_to_save_and_analyze, data_filepath_for_prediction);
    end
    if ~isfield(loaded_capture_data, 'Fs')
        error('Fs not found in captured data file "%s".', data_filepath_for_prediction);
    end
    
    raw_signal_pred = loaded_capture_data.(output_signal_to_save_and_analyze);
    signal_Fs_for_extraction_pred = loaded_capture_data.Fs;
    fprintf('  Loaded captured data. Fs for extraction: %.2f Hz. Signal length: %d.\n', signal_Fs_for_extraction_pred, length(raw_signal_pred));
    
    raw_signal_pred = raw_signal_pred(:)'; % Ensure row vector
catch ME_load_capture
    error('Error loading the captured data file "%s": %s', data_filepath_for_prediction, ME_load_capture.message);
end

if isempty(raw_signal_pred)
    error('No signal data loaded from captured file. Cannot proceed.');
end

% --- Extract Features from Captured Data ---
disp('Extracting features from captured data for K-Means prediction...');
current_features_raw_pred = [];
try
    % Use the same feature extractor, passing the correct Fs and num_dominant_freqs
    current_features_raw_pred = extract_single_signal_features(raw_signal_pred, signal_Fs_for_extraction_pred, num_dominant_freqs_to_use, current_data_source_name_pred, 1); 
catch ME_feat_pred
    error('ERROR during feature extraction from captured data: %s', ME_feat_pred.message);
end

if isempty(current_features_raw_pred)
    error('Feature extraction from captured data returned empty. Cannot proceed.');
end

% --- Preprocess Features (NaN imputation, Scaling) ---
current_features_processed_pred = current_features_raw_pred; 
if ~isempty(feature_imputation_means)
    nan_mask_pred = isnan(current_features_processed_pred);
    if any(nan_mask_pred)
        if length(feature_imputation_means) == length(current_features_processed_pred)
            current_features_processed_pred(nan_mask_pred) = feature_imputation_means(nan_mask_pred);
            disp('  Applied NaN imputation using training means.');
        else
            warning('  Length mismatch for NaN imputation means. Skipping imputation.');
        end
    end
elseif any(isnan(current_features_processed_pred))
     warning('  NaNs in features, but no imputation means. Replacing with 0.');
     current_features_processed_pred(isnan(current_features_processed_pred)) = 0;
end
current_features_processed_pred(isinf(current_features_processed_pred) & current_features_processed_pred > 0) = realmax/10;
current_features_processed_pred(isinf(current_features_processed_pred) & current_features_processed_pred < 0) = -realmax/10;

current_features_scaled_pred = current_features_processed_pred; 
if ~isempty(scaling_zscore_mean) && ~isempty(scaling_zscore_std)
    if length(scaling_zscore_mean) == length(current_features_processed_pred) && ...
       length(scaling_zscore_std) == length(current_features_processed_pred)
        current_features_scaled_pred = (current_features_processed_pred - scaling_zscore_mean) ./ scaling_zscore_std;
        current_features_scaled_pred(isnan(current_features_scaled_pred)) = 0; 
        disp('  Applied Z-score scaling using training parameters.');
    else
        warning('  Length mismatch for Z-score scaling. Using unscaled (but imputed) features.');
    end
else
    disp('  No Z-score scaling parameters loaded. Using unscaled (but imputed) features.');
end
disp('DEBUG: Processed Features from Captured Data:');
disp(current_features_scaled_pred);

% --- K-Means Prediction ---
disp('Performing K-Means prediction on captured data features...');
assigned_cluster_idx_pred = NaN; 
min_dist_pred = Inf;
distances_to_centroids_pred = [];
try
    if isempty(cluster_centroids)
        error('K-Means cluster centroids not loaded.');
    end
    if size(current_features_scaled_pred, 2) ~= size(cluster_centroids, 2)
         error('Mismatch in number of features: current data (%d) vs cluster centroids (%d).', ...
               size(current_features_scaled_pred, 2), size(cluster_centroids, 2));
    end
    if ~isrow(current_features_scaled_pred), current_features_scaled_pred = current_features_scaled_pred(:)'; end

    euclidean_distances_pred = pdist2(current_features_scaled_pred, cluster_centroids, 'euclidean');
    distances_to_centroids_pred = euclidean_distances_pred .^ 2; 
    
    [min_dist_pred, assigned_cluster_idx_pred] = min(distances_to_centroids_pred, [], 2); 
    
    predicted_condition_interpretation_pred = 'Unknown Cluster Assignment';
    if ~isnan(assigned_cluster_idx_pred) && assigned_cluster_idx_pred >= 1 && assigned_cluster_idx_pred <= length(cluster_meanings_derived)
        predicted_condition_interpretation_pred = cluster_meanings_derived(assigned_cluster_idx_pred);
    else
        warning('Assigned cluster index %d is out of bounds or invalid.', assigned_cluster_idx_pred);
    end
catch ME_predict_kmeans_pred
    error('ERROR during K-Means prediction on captured data: %s.', ME_predict_kmeans_pred.message);
end

disp('=====================================================');
fprintf('>>> CAPTURED DATA - K-MEANS PREDICTED CLUSTER: %d\n', assigned_cluster_idx_pred);
fprintf('>>> CAPTURED DATA - INTERPRETED CONDITION: %s\n', predicted_condition_interpretation_pred);
disp('=====================================================');
fprintf('  Distance to assigned cluster centroid (Sq. Euclidean): %.4f\n', min_dist_pred);
if ~isempty(distances_to_centroids_pred)
    disp('  Distances to all centroids (Sq. Euclidean):');
    disp(distances_to_centroids_pred);
end
disp('----------------------------------------------------');

% --- Plotting for Part 2 (Captured Data) ---
figure_handle_pred = figure('Name', 'K-Means Prediction on Captured Data', 'Position', [150 150 1000 600]);
subplot(2,1,1);
time_vector_display_pred = (0:length(raw_signal_pred)-1) / signal_Fs_for_extraction_pred;
plot(time_vector_display_pred, raw_signal_pred); 
xlabel('Time (s)'); ylabel('Signal Value');
interp_char_pred = char(predicted_condition_interpretation_pred); 
title_str_plot_pred = sprintf('Captured Signal: %s | K-Means Cluster: %d (%s)', ...
                             strrep(data_filepath_for_prediction,'_','\_'), ... % Escape underscores for tex
                             assigned_cluster_idx_pred, ...
                             strrep(interp_char_pred,'_','\_'));
title(title_str_plot_pred, 'Interpreter', 'tex'); % Use tex for better underscore handling
grid on;

subplot(2,1,2);
if ~isempty(cluster_centroids) && ~isnan(assigned_cluster_idx_pred) && assigned_cluster_idx_pred > 0 && assigned_cluster_idx_pred <= num_clusters
    bar_data_pred = zeros(1, num_clusters); 
    bar_data_pred(assigned_cluster_idx_pred) = 1; 
    b_pred = bar(bar_data_pred);
    try 
        b_pred.FaceColor = 'flat';
        if assigned_cluster_idx_pred <= size(b_pred.CData,1)
            b_pred.CData(assigned_cluster_idx_pred,:) = [0 0.8 0]; 
        end
    catch
    end
    cluster_names_for_plot_pred = cell(1, num_clusters);
    % CORRECTED SECTION FOR CLUSTER NAMES ON PLOT
    for k_bar_pred = 1:num_clusters
        switch k_bar_pred % k_bar_pred is the cluster index (1-based)
            case 1
                cluster_names_for_plot_pred{k_bar_pred} = 'healthy';
            case 2 % This is for Cluster 2
                cluster_names_for_plot_pred{k_bar_pred} = 'fault at 3';
            case 3 % This is for Cluster 3
                cluster_names_for_plot_pred{k_bar_pred} = 'fault at 5';
            otherwise % For any other clusters (e.g., if num_clusters > 3)
                cluster_names_for_plot_pred{k_bar_pred} = sprintf('Cls %d', k_bar_pred);
        end
    end
    % END OF CORRECTED SECTION
    set(gca, 'XTickLabel', cluster_names_for_plot_pred, 'XTick', 1:num_clusters);
    ylabel('Assignment (1=Assigned)'); ylim([0 1.1]);
    title(sprintf('Assigned to K-Means Cluster %d', assigned_cluster_idx_pred));
else
    cla; text(0.5,0.5, 'K-Means assignment not available for plotting.', 'HorizontalAlignment','center');
end
grid on; 
drawnow; 

disp('--- PART 2: Finished ---');
disp('Master script execution complete.');

