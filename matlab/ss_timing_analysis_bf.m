
addpath('/home/damien/venv_study/ss_timing/code/bayes_factor')

r = -0.4:0.01:0.4;
n = 93;

n_r = length(r);

r_bf = nan(n_r, 2);

r_bf(:, 1) = r;

for i = 1:length(r)
    r_bf(i, 2) = jzs_corbf(r(i), n);
end

save('/home/damien/venv_study/ss_timing/ss_timing_bf.txt', 'r_bf', '-ascii');
