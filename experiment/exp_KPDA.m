function accuracy = exp_KPDA(data, options)

function_path = genpath('../KPDA');
addpath(function_path);


%% prepare data
Xs = data.source_features;
Ys = data.source_labels;
Xl = data.target_labeled_features;
Yl = data.target_labeled_labels;
Xu = data.target_unlabeled_features;
Yu = data.target_unlabeled_labels;
Xt = [Xl Xu];


%% prepare kernels
kparam.kernel_type =  'linear';
[Ks, param_s] = getKernel(Xs, kparam);
[Kt, param_t] = getKernel(Xt, kparam);
Kt_train = getKernel(Xl, Xt, param_t);
Kt_test = getKernel(Xu, Xt, param_t);


%% parameters
param.lambda = options.lambda;
param.gamma = options.gamma;
param.k = options.k;             % subspace base dimension
param.T = options.T;             % #iterations, default=10
param.ks = options.ks;
param.kt = options.kt;
param.P = options.P;
param.Q = options.Q;


%% Laplacian matrices
param.Ls = get_graph(Ks, param.ks);
param.Lt = get_graph(Kt, param.kt);


%% apply SVM to obtain pseudo labels
model = svmtrain(Yl, Xl', '-s 0 -t 0 -q -c 1');
[Yt0, ~, ~] = svmpredict(Yu, Xu', model);


%% run KPDA
accuracy = KPDA(Ks', Kt_train', Kt_test', Ys, Yt0, Yl, Yu, param);
fprintf('KPDA accuracy = %.2f\n', accuracy);
fprintf('===========================');

rmpath(function_path);






