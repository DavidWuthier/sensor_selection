function [xk, f_xk, xk_fig, f_fig, J_fig, H_fig, xnt_fig, t_fig, dnt2_fig] = NewtonEqualityFigure(xk,func,grad,hess,Aeq,beq,opt)
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
% J: gradient of the objective function evaluated in xk;
% H: hessian of the objective function evaluated in xk;
% t: line search in the last newton step of the newton algorithm;
% xnt: search direction in the last step of the newton algorithm;
% dnt2: Squared newton decrement in the last step of the newton algorithm;

if (abs(Aeq*xk-beq)>opt.eps) 
    f_xk = func(xk);
    disp('Initial guess is not feasible');
    return
end

Kn=0;
t=1;
while (Kn<opt.Kn)  
    
% Evaluate the gradient (J) and hessian (H) at xk;
f_xk=func(xk);
J=grad(xk);
H=hess(xk);

%% 1. Descent direction;
% Algorithm 10.3: Solving the KKT system by block elimination;

% step 1;
L = chol(H,'upper'); % Cholesky factoriazation of the hessian, H;

FORMa = L\(L'\Aeq');
FORMb = L\(L'\J);

% step 2 (form the Schur complement, S);
S = -Aeq*FORMa;

% step 3 (determine w - the dual variable);
U = chol(-S,'upper'); % Cholesky factoriazation of the Schur complement;
w = -U\(U'\(Aeq*FORMb));

% step 4 (finale step of algorithm 10.3 - determining the newton step, xnt);
Hv = -Aeq'*w-J;
xnt= L\(L'\Hv); % Descent direction;

dnt2 = xnt'*H*xnt; % Squared newton decrement;

%% 2. Stopping criterion;
if (dnt2/2<opt.eps)
    disp(['stopping criterion met: ', num2str(dnt2/2)]);
    break
end
if (abs(t)<opt.norm)
    disp(['Stopping criterion not met: ', num2str(dnt2/2),' t=',num2str(t)]);
    break
end

Kn=Kn+1; % number of newton iterations;
%% 3. Line search by backtracking;
t=1;
Kb=0;
while (func(xk+t*xnt) > f_xk + opt.alpha*t*J'*xnt)&&(Kb<opt.Kb)
    t=opt.beta*t; % update t;
    Kb=Kb+1; % number of line search iterations;
%     disp(['Kb=',num2str(Kb),' t=',num2str(t), ' criterion=',num2str(func(xk+t*xnt) - ( func(xk) + opt.alpha*t*J'*xnt ))]);
end   

%% 4. Update x;
xk = xk + t*xnt;
f_xk = func(xk);

%% figure output;
xk_fig(:,Kn) = xk;
f_fig(Kn)=func(xk);
J_fig(:,Kn)=grad(xk);
H_fig(:,:,Kn)=hess(xk);
t_fig(Kn) = t;
xnt_fig(:,Kn) = xnt;
w_fig(:,Kn) = w;
dnt2_fig(Kn) = dnt2;

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