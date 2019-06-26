function [xk, f_xk, w, J, H, t, xnt, dnt2] = NewtonEquality(xk,func,grad,hess,Aeq,beq,opt)
%% Input variables;
% xk: initial guess for optimization problem (need to be feasible);
% func: objective function;
% grad: gradient of the objective function;
% hess: hessian of the objective function;
% Aeq and beq: specifies the equality constraint (Aeq*xk=beq);
% opt: struct containing parameters used in the Newton algorithm;

%% Output varibles;
% xk: solution for the equality constrained minimization problem;
% f_xk: the objective function evaluated in xk;
% w: Lagrange multiplier;
% J: gradient of the objective function evaluated in xk;
% H: hessian of the objective function evaluated in xk;
% t: line search in the last newton step of the newton algorithm;
% xnt: search direction in the last step of the newton algorithm;
% dnt2: Squared newton decrement in the last step of the newton algorithm;

if (abs(Aeq*xk-beq)>opt.eps) % Tjeck whether initial guess is feasible;
    f_xk = func(xk);
    disp('Initial guess is not feasible');
    return
end

Kn=0; %Initialize number of newton iterations;
t=1;
while (Kn<opt.Kn) % Iterate until maxim number of iterations is reached; 
    
%% 0. Evaluate the function f_xk, the gradient (J) and hessian (H) at xk;
f_xk=func(xk);
J=grad(xk);
H=hess(xk);

%% 1. Descent direction;
% Algorithm 10.3: Solving the KKT system by block elimination;

% step 1;
L = chol(H,'upper'); % Cholesky factoriazation of the hessian, H;

LAeq = L'\Aeq';
LLJ = L\(L'\J);

% step 2 (form the Schur complement, S);
S = -LAeq'*LAeq;

% step 3 (determine w - the dual variable);
U = chol(-S,'upper'); % Cholesky factoriazation of the Schur complement;
w = -U\(U'\(Aeq*LLJ));

% step 4 (finale step of algorithm 10.3 - determining the newton step, xnt);
Hv = -Aeq'*w-J;
xnt= L\(L'\Hv); % Descent direction;

dnt2 = xnt'*H*xnt; % Squared newton decrement;

%% 2. Stopping criterion;
if (dnt2/2<opt.eps)
    disp(['stopping criterion reached: ', num2str(dnt2/2)]);
    break
end
if (t<opt.norm) % stop if t=0, since this implies no update of xk;
    disp(['Stopping criterion not reached: ', num2str(dnt2/2),' t=0']);
    break
end

Kn=Kn+1; % number of newton iterations;
%% 3. Line search by backtracking;
t=1;
Kb=0;
while (func(xk+t*xnt) > f_xk + opt.alpha*t*J'*xnt)&&(Kb<opt.Kb) % Iterate until backtracking criterion or maximum number of iterations is reached;
    t=opt.beta*t; % update t;
    Kb=Kb+1; % number of line search iterations;
end   

%% 4. Update x;
xk = xk + t*xnt;

%% Print;
disp('-----------------------------------------------------------------------');
disp(['Number of newton iterations, Kn=', num2str(Kn)]);
disp(['Number of line search iterations, Kb=', num2str(Kb),', t=', num2str(t)]);
disp(['Evaluate stopping criterion, dnt2/2=', num2str(dnt2/2)]);
disp(['f(xk)=', num2str(f_xk)]);
disp('-----------------------------------------------------------------------');
end

if Kn == opt.Kn
    disp(['Maximum number of iterations reached: ', num2str(Kn), ' iterations']);
    return
end

end