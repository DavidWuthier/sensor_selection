%% Clear memory;
clear all; clc; beep off;

rng(34);

par.m = 40; % number of linear measurements;
par.n = 2; % dimensions of x;
par.k = 4; % subset of m that minimizes the volume;;

par.kappa = 0.01*par.n/par.m;

% par.a = rand(par.n,par.m);
% Ap = par.a;

load('Aprime.mat');
par.a=Aprime;

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

eps = 0.001;
g_z0_num = derivative.num_grad(func,z0,eps);
h_z0_num = derivative.num_hess(func,z0,eps);

det_hes_num=det(h_z0_num);
eig_min_num=min(eig(h_z0_num),[],1);

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

opt.eps   = 1e-12; % stopping criterion;
opt.norm  = 1e-12; % stopping criterion for search direction;

tic;
[zk, f_zk, w, J_zk, H_zk, t, xnt, dnt2] = NewtonEquality(z0,func,grad,hess,A,b,opt); % Newton algorithm;
toc;

aa = J_zk + A'*w;

%% Compare to matlab's solver (fmincon);
CSxk = A*zk-b; % tjeck constraints;
CSx0 = A*z0-b; % initial guess;

% [z_min, f_min] = fmincon(func,z0,[],[],A,b);
% disp(['xk^T=(',num2str(zk'),'), ','f(xk)=', num2str(f_zk)]);
% disp(['x_min=(',num2str(z_min'),'), ','f(x_min)=', num2str(f_min)]);
% 
% z_diff = zk - z_min;
% f_diff = f_zk - f_min;
% 
% disp(['x_diff=(',num2str(z_diff'),'), ','f_diff=', num2str(f_diff)]);
