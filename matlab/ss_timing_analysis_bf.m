
addpath('/home/damien/venv_study/ss_timing/code/bayes_factor')

r = -0.5:0.01:0.5;
n = 93;

bf10 = [];

for i = 1:length(r)
    bf10 = [bf10, jzs_corbf(r(i), n)];
end

save('/home/damien/venv_study/ss_timing/ss_timing_bf.txt', 'bf10', '-ascii');
