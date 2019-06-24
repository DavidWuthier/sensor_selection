
close all
clear
clc

% syms ts ptx pty psx psy
% 
% t = atan((-sin(ts)*(ptx - psx) + cos(ts)*(pty - psy))/...
%          (cos(ts)*(ptx - psx) + sin(ts)*(pty - psy)));
% 
% h1 = matlabFunction(diff(t,ptx), 'File', 'h1');
% h2 = matlabFunction(diff(t,pty), 'File', 'h2');

ns = 40;
ptx = 0.1;
pty = 0.0;

p = (2*pi)/ns*(0:ns-1);

% psx = (1 + p).*cos(p);
% psy = (1 + p).*sin(p);
psx = cos(p);
psy = sin(p);
psx(1:10) = 0.5*psx(1:10);
psy(1:10) = 0.5*psy(1:10);
ts = p + pi;
A = zeros(ns,2);
a = zeros(2,2,ns);

for k = 1:ns
    A(k,1) = h1(psx(k), psy(k), ptx, pty, ts(k));
    A(k,2) = h2(psx(k), psy(k), ptx, pty, ts(k));
    a(:,:,k) = A(k,:).' * A(k,:);
end

k = 4;

cvx_begin
    variable z(ns)
    expression s(2, 2, ns)
    for i = 1:ns
        s(:,:,i) = z(i) * A(i,:).' * A(i,:);
    end
    maximize(log_det(sum(s, 3)))
    subject to
        ones(1,ns)*z == k
        0 <= z <= 1
cvx_end

figure
quiver(psx, psy, cos(ts), sin(ts), 0.2)
hold on
grid on
axis square
plot(ptx, pty, 'ro')
quiver(psx, psy, z.'.*cos(ts), z.'.*sin(ts), 0.2, 'r')

