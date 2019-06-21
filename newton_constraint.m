%% Clear memory;
clear all; clc; beep off;

%% Initialize parameters;
% Parameters describing the quadratic function;

par.P = [ 3  1 1;
          1  2 1;
          2  2 3];
par.q = [-2 -1 0]';
par.r = 5;

PD = min(eig(par.P),[],1); %Convexity tjeck (all eigenvalues should be nonnegative);

% Parameters describing equality constraints;
par.A = [1  1 1;
         1 -1 0];
par.b = [5 0];

x0 = [2 2 1]'; % Initial guess;

if (par.A*x0~=par.b) 
    disp('Initial guess is not feasible');
    return
end

% Parameters used in the newton algorithm;
par.Kn = 1000; % maximal number of newton iterations;
par.Kb = 100; % maximal number of line search iterations;
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

% Algorithm 10.3: Solving the KKT system by block elimination (Hessian, H, is positive definite);
% step 1;
L = chol(H,'upper'); % Cholesky factoriazation of the hessian, H;

FORMa = L\(L'\par.A');
FORMb = L\(L'\J);

% step 2 (form the Schur complement, S);
S = -par.A*FORMa;

% step 3 (determine w);
U = chol(-S,'upper'); % Cholesky factoriazation of the Schur complement;
w  = -U\(U'\(par.A*FORMb));

% step 4 (finale step of algorithm 10.3: determines v);
Hv = -par.A'*w-J;
xnt  = L\(L'\Hv); % Descent direction;

S1  = -par.A*(H^-1)*par.A';
Sw1 = par.A*(H^-1)*J;
w1  = (S1^-1)*Sw1;
Hv1 = -par.A'*w1-J;
xnt= (H^-1)*Hv1;


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
%     disp(['Kb=',num2str(Kb),' t=',num2str(t), ' criterion=',num2str(func(xk+t*xnt) - ( func(xk) + par.alpha*t*J'*xnt ))]);
end

%% 4. Update x;
xk = xk + t*xnt;
f_xk = func(xk);

%% Print;
disp('-----------------------------------------------------------------------');
disp(['Number of newton iterations, Kn=', num2str(Kn)]);
disp(['Number of line search iterations, Kb=', num2str(Kb),', t=', num2str(t)]);
disp(['Evaluate stopping criterion, dnt2/2=', num2str(dnt2/2)]);
disp(['xk^T=(',num2str(xk'),'), ','f(xk)=', num2str(f_xk)]);
disp('-----------------------------------------------------------------------');
end

CSxk = par.A*xk-par.b; % tjeck constraint;
CSx0 = par.A*x0-par.b;

[x_min, f_min] = fmincon(func,x0,[],[],par.A,par.b);
disp(['x_min=(',num2str(xk'),'), ','f(x_min)=', num2str(f_min)]);
