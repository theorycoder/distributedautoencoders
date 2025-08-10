%% privacy_results.m
% MATLAB script version of privacy_results.py
clear; clc; tic;

% Epsilon values
eps_vals = [0.01, 0.1, 1.0, 10, 20];
ne = numel(eps_vals);

% Preallocate
FM_noisy2     = zeros(1, ne);
FM_noiseless  = zeros(1, ne);
dpsgd2        = zeros(1, ne);
dpsgd3        = zeros(1, ne);
nonprivate    = zeros(1, ne);
nonprivate_time = zeros(1, ne);
FM_time       = zeros(1, ne);
dpsgd_time    = zeros(1, ne);
PALM_noisy2   = zeros(1, ne);
PALM_noiseless= zeros(1, ne);
PALM_time     = zeros(1, ne);

% --- Load FM accuracy ---
for i = 1:ne
    FM_noiseless(i) = read_mean(sprintf('results/FMaccuracy_noislessInp_%d.txt', i-1));
    FM_noisy2(i)    = read_mean(sprintf('results/FMaccuracy_noisyInp_%d_1_5.txt', i-1));
end

% --- Load DPSGD accuracy ---
for i = 1:ne
    dpsgd2(i) = read_mean(sprintf('results/dpsgdaccuracy_%d_1_0_1.txt', i-1));
    dpsgd3(i) = read_mean(sprintf('results/dpsgdaccuracy_%d_1_1_5.txt', i-1));
end

% --- Load Non-private ---
for i = 1:ne
    nonprivate(i) = read_mean('results/nonprivate_0.txt');
end

% --- Time data ---
for i = 1:ne
    nonprivate_time(i) = read_mean(sprintf('results/nonPrivate_time_%d_0_0_1.txt', i-1));
    FM_time(i)         = read_mean(sprintf('results/fm_time_%d_1_0_1.txt', i-1));
    dpsgd_time(i)      = read_mean(sprintf('results/dpsgd_time_%d_1_0_1.txt', i-1));
end

% --- PALM ---
for i = 1:ne
    PALM_noiseless(i) = read_mean(sprintf('results/PALMaccuracy_noislessInp_%d.txt', i-1));
    PALM_noisy2(i)    = read_mean(sprintf('results/PALMaccuracy_noisyInp_%d_1_5.txt', i-1));
    PALM_time(i)      = read_mean(sprintf('results/PALM_time_%d_1_0_1.txt', i-1));
end

% --- Debug prints ---
diff1 = (mean(PALM_noiseless, 2) - mean(dpsgd2, 2)) ./ mean(dpsgd2, 2) * 100;
fprintf('diff1: %.6f\n', diff1);
diff2 = (mean(PALM_noisy2, 2) - mean(dpsgd3, 2)) ./ mean(dpsgd3, 2) * 100;
fprintf('diff2: %.6f\n', diff2);
fprintf('dpsgd_time: %s\n', mat2str(dpsgd_time, 6));
fprintf('PALM_time: %s\n', mat2str(PALM_time, 6));
time_diff1 = (mean(dpsgd_time, 2) - mean(PALM_time, 2)) ./ mean(dpsgd_time, 2) * 100;
fprintf('time_diff1: %.6f\n', time_diff1);

% --- Plot ---
figure('Units','pixels','Position',[100 100 800 600]); hold on;
xidx = 1:ne;
plot(xidx, nonprivate, '-.o', 'LineWidth', 3, 'MarkerSize', 10);
plot(xidx, FM_noiseless, '-.*', 'LineWidth', 3, 'MarkerSize', 10);
plot(xidx, FM_noisy2, '-.d', 'LineWidth', 3, 'MarkerSize', 10);
plot(xidx, dpsgd2, '-.s', 'LineWidth', 3, 'MarkerSize', 10);
plot(xidx, dpsgd3, '-.x', 'LineWidth', 3, 'MarkerSize', 10);
plot(xidx, PALM_noiseless, '--^', 'LineWidth', 3, 'MarkerSize', 10);
plot(xidx, PALM_noisy2, '--h', 'LineWidth', 3, 'MarkerSize', 10);
hold off; ylim([0.37 1]); grid on;

ax = gca;
ax.XTick = xidx;
ax.XTickLabel = arrayfun(@(v) num2str(v), eps_vals, 'UniformOutput', false);
set(ax, 'XDir', 'reverse');

ylabel('accuracy', 'FontSize', 16, 'Interpreter','none');
xlabel('privacy budget $\epsilon$', 'FontSize', 16, 'Interpreter','latex');

legend(...
    {'non-private (standard BP)', ...
     'FM noiseless inputs (BK)', ...
     'FM noisy inputs, $\sigma=5$ (BK)', ...
     'DP-SGD noiseless inputs (BK+GC)', ...
     'DP-SGD noisy inputs, $\sigma=5$ (BK+GC)', ...
     'SPOF noiseless inputs (BK)', ...
     'SPOF noisy inputs, $\sigma=5$ (BK)'}, ...
     'Interpreter','latex', ...
     'Location','southwest');

if ~exist('results', 'dir'); mkdir('results'); end
print(gcf, fullfile('results','results'), '-dpdf', '-fillpage');

fprintf('Done in %.2f seconds\n', toc);

%% Helper function
function m = read_mean(fname)
    if ~isfile(fname)
        error('File not found: %s', fname);
    end
    data = readmatrix(fname);
    data = data(~isnan(data));
    if isempty(data)
        m = NaN;
    else
        m = mean(data(:));
    end
end

