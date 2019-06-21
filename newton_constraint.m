%% Clear memory;
clear all; clc; beep off;

%% Initialize parameters;
% Parameters describing the quadratic function;

par.P = [ 0.03  0.01;
          0.01  0.02];
par.q = [-2 -1]';
par.r = 5;

% Parameters describing equality constraints;
par.A = [1 1];
par.k = 5;

x0 = [5 0]'; % Initial guess;

if (par.A*x0~=par.k) 
    disp('Initial guess is not feasible');
    return
end

% Parameters used in the newton algorithm;
par.Kn = 20; % maximal number of newton iterations;
par.Kb = 20; % maximal number of line search iterations;
par.alpha = 0.25;
par.beta  = 0.50; 
par.eps   = 1e-12; % tolerance level;

%% Initialize objective function and the corresponding gradient and hessian;
func = @(x) fun.quadratic(x,par);
grad = @(x) fun.quadratic_grad(x,par);
hess = @(x) fun.quadratic_hess(x,par);

xk=x0;
Kn=0;
while Kn<par.Kn
% Evaluate the gradient (J) and hessian (H) at xk;
J=grad(xk);
H=hess(xk);

%% 1. Descent direction;
% xnt = -H\J; % Descent direction;
% dnt = xnt'*(H\xnt); % Newton decrement;

L = chol(H,'upper'); %Cholesky factoriazation of the hessian, H;

xnt = -L\(L'\J); % Descent direction;
dnt2 = xnt'*(L\(L'\xnt)); % Squared newton decrement;

%% 2. Stopping criterion;
if dnt2/2<par.eps
    break
end
Kn=Kn+1; % number of newton iterations;

%% 3. Line search by backtracking;
t=1;
Kb=0;
while (func(xk+t*xnt) > func(xk) + par.alpha*t*J'*xnt)&&(Kb<par.Kb)
    t=par.beta*t; % update t;
    Kb=Kb+1; % number of line search iterations;
end

%% 4. Update x;
xk = xk + t*xnt;
f_xk = func(xk);

%% Print;
disp(['----------------------------------------------------------------------']);
disp(['Number of newton iterations, Kn=', num2str(Kn)]);
disp(['Number of line search iterations, Kb=', num2str(Kb),', t=', num2str(t)]);
disp(['Evaluate stopping criterion, dnt2/2=', num2str(dnt2/2)]);
disp(['xk^T=(',num2str(xk'),'), ','f(xk)=', num2str(f_xk)]);
disp(['----------------------------------------------------------------------']);
end

H_eig = eig(H); %Convexity tjeck (all eigenvalues should be nonnegative);

% [x_min, f_min] = fminunc(func,x0);
% disp(['x_min=(',num2str(xk'),'), ','f(x_min)=', num2str(f_min)]);
