function accuracy = KPDA(Xs, Xl, Xu, Ys, Yt0, Yl, Yu, options)

%% data
Xt = [Xl Xu];
ns = size(Xs,2);
nt = size(Xt,2);
C = length(unique(Ys));
Ls = options.Ls;
Lt = options.Lt;
P = options.P;
Q = options.Q;


%% parameters
lambda = options.lambda;
gamma = options.gamma;
max_iter = options.T;

opts.record = 0; %
opts.mxitr  = 1000;
opts.xtol = 1e-3;
opts.gtol = 1e-3;
opts.ftol = 1e-5;



E = gamma * Ls - lambda * eye(ns);
E = Xs * E * Xs';
F = gamma * Lt - lambda * eye(nt);
F = Xt * F * Xt';


for t = 1:max_iter
    
    % Construct MMD matrix
    [Ms, Mt, Mst, Mts] = constructMMD(ns,nt,Ys,Yt0,C);
    
    Ts = Xs*Ms*Xs';
    Tt = Xt*Mt*Xt';
    Tst = Xs*Mst*Xt';
    Tts = Xt*Mts*Xs';
    
    V_S = Ts + E;
    U_ST = Tst;
    U_TS = Tts;
    V_T = Tt + F;
    
    %% fix P
    [new_Q, ~, ~]= OptStiefelGBB(Q, @funQ, opts, V_T, U_TS, U_ST, P);
    
    %% fix Q
    [new_P, ~, ~]= OptStiefelGBB(P, @funP, opts, V_S, U_TS, U_ST, new_Q);
    
    
    %% latent target features
    tilde_Xl = Xl' * new_Q;
    tilde_Xu = Xu' * new_Q;
    
    
    Xtrain = [Xl' tilde_Xl];
    Ytrain = Yl;
    Xtest = [Xu' tilde_Xu];
    Ytest = Yu;
    
    
    fprintf('=========== %d iteration =========== \n', t);
    model = svmtrain(Ytrain, Xtrain, '-s 0 -t 0 -q -c 1');
    [Yt0, result_acc, ~] = svmpredict(Ytest, Xtest, model);
    accuracy = result_acc(1);
    
    P = new_P;
    Q = new_Q;
    
end

end

function [F, G] = funQ(X, V_T, U_TS, U_ST, P)
G = 2 * V_T * X + (U_TS + U_ST') * P;
F = trace(X' * V_T * X) + trace(X' * (U_TS + U_ST') * P);
end

function [F, G] = funP(X, V_S, U_TS, U_ST, Q)
G = 2 * V_S * X + (U_TS' + U_ST) * Q;
F = trace(X' * V_S * X) + trace(Q' * (U_TS + U_ST') * X);
end





