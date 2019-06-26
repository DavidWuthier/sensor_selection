%% Clear memory;
clear all; clc; beep off;

rng(34);

par.m = 400; % number of linear measurements;
par.n = 10; % dimensions of x;
par.k = 40; % subset of m that minimizes the volume;;

par.kappa = 0.01*par.n/par.m;

par.a = rand(par.n,par.m);
Ap = par.a;

% load('Aprime.mat');
% par.a=Aprime;

z0 = par.k/par.m * ones(par.m,1);

%% Initialize objective function and the corresponding gradient and hessian;
obj = @(x) fun.LogVolume(x.par);

func = @(x) -fun.ApproxLogVolume(x,par);
grad = @(x) -fun.ApproxLogVolume_grad(x,par);
hess = @(x) -fun.ApproxLogVolume_hess(x,par);

f_z0 = func(z0);
g_z0 = grad(z0);
h_z0 = hess(z0);

det_hes=det(h_z0);
eig_min=min(eig(h_z0),[],1);

% eps = 0.001;
% g_z0_num = derivative.num_grad(func,z0,eps);
% h_z0_num = derivative.num_hess(func,z0,eps);
% 
% det_hes_num=det(h_z0_num);
% eig_min_num=min(eig(h_z0_num),[],1);

%% Parameters describing equality constraints (Ax=b);
A = ones(1,par.m);
b = par.k;

b0 = A*z0;
%% Run Newton algorithm for equality constrained minimization;

% Parameters used in the newton algorithm;
opt.Kn = 500; % maximal number of newton iterations;
opt.Kb = 100; % maximal number of line search iterations;

opt.alpha = 0.1; % alpha in (0.0; 0.5)
opt.beta  = 0.50; % beta in (0.5; 1.0)

opt.eps   = 1e-6; % stopping criterion;
opt.norm  = 1e-6; % stopping criterion for search direction;

% tic;
% [zk, f_zk, w, J_zk, H_zk, t, xnt, dnt2] = NewtonEquality(z0,func,grad,hess,A,b,opt); % Newton algorithm;
% toc;
% 
% CSzk = A*zk-b; % tjeck constraints;
CSz0 = A*z0-b; % initial guess;


%% Compare to matlab's solver (fmincon);

% [z_min, f_min] = fmincon(func,z0,[],[],A,b);
% disp(['xk^T=(',num2str(zk'),'), ','f(xk)=', num2str(f_zk)]);
% disp(['x_min=(',num2str(z_min'),'), ','f(x_min)=', num2str(f_min)]);
% 
% z_diff = zk - z_min;
% f_diff = f_zk - f_min;
% 
% disp(['x_diff=(',num2str(z_diff'),'), ','f_diff=', num2str(f_diff)]);

%% Figures
[zk, f_zk, fig_zk, fig_f, fig_J, fig_H, fig_znt, fig_t, fig_dnt2] = NewtonEqualityFigure(z0,func,grad,hess,A,b,opt); % Newton algorithm;

max_search = fig_t.*fig_znt(6,:);

iterations = 0:size(fig_zk,2);

fig_zk = [z0 fig_zk];
fig_f = [f_z0 fig_f];

max_zk = fig_zk(6,:);
%%
subplot(2,2,1);
semilogx(fig_f);   
title('Objective function (log)')
ylabel('f(z^k)') 
xlabel('Newton iterations (incl. initial guess), k') 
subplot(2,2,2);
plot(max_zk);
title('Choice probability')
ylabel('z^{k}')
xlabel('Newton iterations (incl. initial guess), k')
subplot(2,2,3);
semilogx(fig_dnt2);
title('Stopping criterion (log)')
ylabel('\lambda^2(z^k)/2') 
xlabel('Newton iterations, k') 
subplot(2,2,4);
plot(max_search);
title('Search direction')
ylabel('t\Deltaz^k')
xlabel('Newton iterations, k')

print -deps NewtonFig

