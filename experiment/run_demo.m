clc;
clear;

addpath('common_tool/');

dataname = 'wireless';
source_domain = 'es';
target_domain = 'en';


%% parameter
options.k = 10;
options.ks = 8;
options.kt = options.ks;
options.lambda = 100;
options.gamma = 10;


%% run experiment
for trial = 1:5
    
    %% data
    data = prepare_data(dataname, source_domain, target_domain, trial);
    
    fprintf('%s: %d-th trial running ... \n', dataname, trial);
    fprintf('task: %s -> %s  \n', source_domain, target_domain);
    
    fprintf('++++++++++++++ init P and Q ++++++++++++++ \n');
    load('init.mat');
    
    options.P = P;
    options.Q = Q;
    options.T = 10;
    
    fprintf(1, 'k=%d, ks=%d, kt=%d, lambda=%.3f, gamma=%.3f \n', options.k, options.ks, options.kt, options.lambda, options.gamma);
    
    acc(trial) = exp_KPDA(data, options);
end
fprintf('Kernelized KPDA acc: %.2f (%.2f) \n', mean(acc), std(acc));

rmpath('common_tool/');



