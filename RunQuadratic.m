%% Clear memory;
clear all; clc; beep off;

%% Initialize parameters describing the quadratic objective function;
par.P = [ 3  1  1;
          1  2  1;
          1  1  1];
      
par.q = [-1 -2 -3]';
par.r = 5;

% Convexity tjeck (all eigenvalues should be nonnegative);
PD = min(eig(par.P),[],1); 

%% Initialize objective function and the corresponding gradient and hessian;
func = @(x) fun.quadratic(x,par);
grad = @(x) fun.quadratic_grad(x,par);
hess = @(x) fun.quadratic_hess(x,par);

%% Parameters describing equality constraints (Ax=b);
A = [ 1  1  1];
b = 12;

%% Run Newton algorithm for equality constrained minimization;

% Parameters used in the newton algorithm;
opt.Kn = 500; % maximal number of newton iterations;
opt.Kb = 100; % maximal number of line search iterations;

opt.alpha = 0.25; % alpha in (0.0; 0.5)
opt.beta  = 0.50; % beta in (0.5; 1.0)

opt.eps   = 1e-12; % stopping criterion;
opt.norm  = 1e-12; % stopping criterion for search direction;

x0 = [4 4 4]'; % Initial guess;
[xk, f_xk, J, H] = NewtonEquality(x0,func,grad,hess,A,b,opt); % Newton algorithm;

%% Compare to matlab's solver (fmincon);
CSxk = A*xk-b; % tjeck constraints;
CSx0 = A*x0-b; % initial guess;

[x_min, f_min] = fmincon(func,x0,[],[],A,b);
disp(['xk^T=(',num2str(xk'),'), ','f(xk)=', num2str(f_xk)]);
disp(['x_min=(',num2str(x_min'),'), ','f(x_min)=', num2str(f_min)]);

x_diff = xk - x_min;
f_diff = f_xk - f_min;

disp(['x_diff=(',num2str(x_diff'),'), ','f_diff=', num2str(f_diff)]);
