%% Clear memory;
clear all; clc; beep off;

%% Initialize parameters;
% Parameters describing the quadratic function;
n=3;
 
par.P = [ 3  1  1;
          1  2  1;
          1  1  1];
      
par.q = [-1 -2 -3]';
par.r = 5;

PD = min(eig(par.P),[],1); % Convexity tjeck (all eigenvalues should be nonnegative);

% Parameters describing equality constraints;
par.A = [ 1  1  1];
par.b = [12]';

x0 = [4 4 4]'; % Initial guess;

if (par.A*x0~=par.b) 
    disp('Initial guess is not feasible');
    return
end

% Parameters used in the newton algorithm;
par.Kn = 500; % maximal number of newton iterations;
par.Kb = 100; % maximal number of line search iterations;
par.alpha = 0.25;
par.beta  = 0.50; 
par.eps   = 1e-12; % tolerance level;
par.norm  = 1e-12;

%% Initialize objective function and the corresponding gradient and hessian;
func = @(x) fun.quadratic(x,par);
grad = @(x) fun.quadratic_grad(x,par);
hess = @(x) fun.quadratic_hess(x,par);

f_x0 = func(x0);

xk=x0;
Kn=0;
norm=1;
while (Kn<par.Kn)
% Evaluate the gradient (J) and hessian (H) at xk;
J=grad(xk);
H=hess(xk);

%% 1. Descent direction;
% xnt = -H\J; % Descent direction;
% dnt = xnt'*(H\xnt); % Newton decrement;

% Algorithm 10.3: Solving the KKT system by block elimination (Hessian, H, is positive definite);
% step 1;
L = chol(H,'upper'); % Cholesky factoriazation of the hessian, H;

FORMa = L\(L'\par.A');
FORMb = L\(L'\J);

% step 2 (form the Schur complement, S);
S = -par.A*FORMa;

% step 3 (determine w - the dual variable);
U = chol(-S,'upper'); % Cholesky factoriazation of the Schur complement;
w  = -U\(U'\(par.A*FORMb));

% step 4 (finale step of algorithm 10.3 - determining the newton step, xnt);
Hv = -par.A'*w-J;
xnt= L\(L'\Hv); % Descent direction;

dnt2 = xnt'*H*xnt; % Squared newton decrement;

% Tjeck matrix decomposition;
% a = (H^-1)*par.A';
% b = (H^-1)*J;
% S1 = -par.A*a;
% w1 = (S1^-1)*par.A*b;
% Hv1= -par.A'*w1-J;
% v1 = (H^-1)*(Hv1);
%% 2. Stopping criterion;
if (dnt2/2<par.eps)
    f_xk = func(xk);
    disp(['stopping criterion met: ', num2str(dnt2/2)]);
    break
end
if (abs(norm)<par.norm)
    disp(['Stopping criterion not met: ', num2str(dnt2/2),' t=',num2str(t)]);
    break
end

Kn=Kn+1; % number of newton iterations;
%% 3. Line search by backtracking;
t=1;
Kb=0;
while (func(xk+t*xnt) > func(xk) + par.alpha*t*J'*xnt)&&(Kb<par.Kb)
    t=par.beta*t; % update t;
    Kb=Kb+1; % number of line search iterations;
%     disp(['Kb=',num2str(Kb),' t=',num2str(t), ' criterion=',num2str(func(xk+t*xnt) - ( func(xk) + par.alpha*t*J'*xnt ))]);
end

%% 4. Update x;
xk = xk + t*xnt;

norm=t; % Used as an additional stopping criterion (stop if t close to zero);

f_xk = func(xk);
%% Print;
disp('-----------------------------------------------------------------------');
disp(['Number of newton iterations, Kn=', num2str(Kn)]);
disp(['Number of line search iterations, Kb=', num2str(Kb),', t=', num2str(t)]);
disp(['Evaluate stopping criterion, dnt2/2=', num2str(dnt2/2)]);
disp(['xk^T=(',num2str(xk'),'), ','f(xk)=', num2str(f_xk)]);
disp('-----------------------------------------------------------------------');
end

CSxk = par.A*xk-par.b; % tjeck constraints;
CSx0 = par.A*x0-par.b; % initial guess;

[x_min, f_min] = fmincon(func,x0,[],[],par.A,par.b);
disp(['xk^T=(',num2str(xk'),'), ','f(xk)=', num2str(f_xk)]);
disp(['x_min=(',num2str(x_min'),'), ','f(x_min)=', num2str(f_min)]);

x_diff = xk - x_min;
f_diff = f_xk - f_min;

disp(['x_diff=(',num2str(x_diff'),'), ','f_diff=', num2str(f_diff)]);
