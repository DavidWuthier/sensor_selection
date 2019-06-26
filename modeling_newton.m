beep off
close all
clear
clc

rng(34);

par.m = 100; % number of linear measurements;
par.n = 1; % dimensions of x;
par.k = 5; % subset of m that minimizes the volume;;

par.kappa = 0.01*par.n/par.m;

z0 = par.k/par.m * ones(par.m,1);

% % # of sensor
% m = 40;
% % # of target points
% np = 1;
% Time
t = 0:(2*pi)/par.n:2*pi*(1-1/par.n);
% # target points
ptx = 0.1;% 0.2*cos(t);
pty = 0;% 0.2*sin(2*t);
% Azimut of cameras
p = (2*pi)/par.m*(0:par.m-1);
% Camera positions
psx = cos(p);
psy = sin(p);
% Camera angles
ts = p + pi;

% Jacobian of the measurements
A = zeros(par.n,2*par.n,par.m);
for i = 1:par.m
    for j = 1:par.n
        A(j,2*j-1,i) = fun.h1(psx(i), psy(i), ptx(j), pty(j), ts(i));
        A(j,2*j,i) = fun.h2(psx(i), psy(i), ptx(j), pty(j), ts(i));
    end
end

Aprime(:,:) = A(1,:,:);
save('Aprime','Aprime');

par.a(:,:) = A(1,:,:);

% k = 4;
par.kappa = 0.01 *2/par.m;

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
Aeq = ones(1,par.m);
beq = par.k;

beq0 = Aeq*z0;
%% Run Newton algorithm for equality constrained minimization;

% Parameters used in the newton algorithm;
opt.Kn = 500; % maximal number of newton iterations;
opt.Kb = 100; % maximal number of line search iterations;

opt.alpha = 0.1; % alpha in (0.0; 0.5)
opt.beta  = 0.50; % beta in (0.5; 1.0)

opt.eps   = 1e-6; % stopping criterion;
opt.norm  = 1e-6; % stopping criterion for search direction;

tic;
[z, f_z, w, J_zk, H_zk, t, xnt, dnt2] = NewtonEquality(z0,func,grad,hess,Aeq,beq,opt); % Newton algorithm;
z_newton = z;
toc;

% CSzk = Aeq*z-beq; % tjeck constraints;
CSz0 = Aeq*z0-beq; % initial guess;

tic;
cvx_begin
    variable z(par.m)
    expressions s1(2*par.n,2*par.n,par.m) s2(par.m)
    
    for i = 1:par.m
        s1(:,:,i) = z(i) * A(:,:,i).' * A(:,:,i);
        s2(i) = log(z(i)) + log(1 - z(i));
    end
    
    maximize(log_det(sum(s1, 3)) + par.kappa*sum(s2))
    
    subject to
        ones(1,par.m)*z == par.k
cvx_end
z_CVX = z;
toc;
tjeck = [z_newton z_CVX z_newton-z_CVX];

s1 = fun.cov(z,par);

ps = log(det(s1))
pwc = ps + 2*par.m*par.kappa

zs = sort(z);
zl = z >= zs(end-par.k+1);

s1 = fun.cov(zl,par);

psl = log(det(s1))

e = (psl - pwc)/pwc

figure
quiver(psx, psy, cos(ts), sin(ts))
hold on
grid on
axis square
plot(ptx, pty, 'ro')
quiver(psx, psy, z.'.*cos(ts), z.'.*sin(ts), 'r')
plot(psx(zl), psy(zl), 'ko')

%% Figures (convergence);
[zk, f_zk, fig_zk, fig_f, fig_J, fig_H, fig_znt, fig_t, fig_dnt2] = NewtonEqualityFigure(z0,func,grad,hess,Aeq,beq,opt); % Newton algorithm;

max_search = fig_t.*fig_znt(6,:);

iterations = 0:size(fig_zk,2);

fig_zk = [z0 fig_zk];
fig_f = [f_z0 fig_f];

max_zk = fig_zk(6,:);

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

obj_newton = func(z_newton);
obj_CVX = func(z_CVX);
