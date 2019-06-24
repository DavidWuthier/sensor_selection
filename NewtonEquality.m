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
norm=1;
while (Kn<opt.Kn)

if (abs(Aeq*xk-beq)>opt.eps) 
    f_xk = func(xk);
    disp('equality constraint is not met');
    return
end    
    
% Evaluate the gradient (J) and hessian (H) at xk;
J=grad(xk);
H=hess(xk);

%% 1. Descent direction;
% Algorithm 10.3: Solving the KKT system by block elimination;

% step 1;
Q = eye(size(Aeq,1),size(Aeq,1));

UL = H + Aeq'*Q*Aeq; 
% UR = J + Aeq'*Q*Aeq;

L = chol(H,'upper'); % Cholesky factoriazation of the hessian, H;

FORMa = L\(L'\Aeq');
FORMb = L\(L'\J);

% step 2 (form the Schur complement, S);
S = -Aeq*FORMa;

% step 3 (determine w - the dual variable);
U = chol(-S,'upper'); % Cholesky factoriazation of the Schur complement;
w  = -U\(U'\(Aeq*FORMb));

% step 4 (finale step of algorithm 10.3 - determining the newton step, xnt);
Hv = -Aeq'*w-J;
xnt= L\(L'\Hv); % Descent direction;

dnt2 = xnt'*H*xnt; % Squared newton decrement;

%% 2. Stopping criterion;
if (dnt2/2<opt.eps)
    f_xk = func(xk);
    disp(['stopping criterion met: ', num2str(dnt2/2)]);
    break
end
if (abs(norm)<opt.norm)
    f_xk = func(xk);
    disp(['Stopping criterion not met: ', num2str(dnt2/2),' t=',num2str(t)]);
    break
end

Kn=Kn+1; % number of newton iterations;
%% 3. Line search by backtracking;
t=1;
Kb=0;
while (func(xk+t*xnt) > func(xk) + opt.alpha*t*J'*xnt)&&(Kb<opt.Kb)
    t=opt.beta*t; % update t;
    Kb=Kb+1; % number of line search iterations;
    
%     disp(['Kb=',num2str(Kb),' t=',num2str(t), ' criterion=',num2str(func(xk+t*xnt) - ( func(xk) + opt.alpha*t*J'*xnt ))]);
end

    if max(xk+t*xnt,[],1)>1 
        f_xk = func(xk);
        disp('Out of unit interval, xk>1');
        return
    end
    if min(xk+t*xnt,[],1)<0
        f_xk = func(xk);
        disp('Out of unit interval, xk<0');
        return
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
disp(['f(xk)=', num2str(f_xk)]);
disp('-----------------------------------------------------------------------');
end

    if Kn == opt.Kn
        disp(['Maximum number of iterations reached: ', num2str(Kn), ' iterations']);
        return
    end

end