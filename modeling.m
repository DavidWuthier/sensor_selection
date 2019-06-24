
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

m = 40;
ptx = 0.1;
pty = 0.0;

p = (2*pi)/m*(0:m-1);

psx = cos(p);
psy = sin(p);
ts = p + pi;
A = zeros(m,2);
a = zeros(2,2,m);

for k = 1:m
    A(k,1) = h1(psx(k), psy(k), ptx, pty, ts(k));
    A(k,2) = h2(psx(k), psy(k), ptx, pty, ts(k));
    a(:,:,k) = A(k,:).' * A(k,:);
end

k = 4;
h = 0.01 *2/m;

tic
cvx_begin
    variable z(m)
    expressions s1(2, 2, m) s2(m)
    
    for i = 1:m
        s1(:,:,i) = z(i) * A(i,:).' * A(i,:);
        s2(i) = log(z(i)) + log(1 - z(i));
    end
    
    maximize(log_det(sum(s1, 3)) + h*sum(s2))
    
    subject to
        ones(1,m)*z == k
        0 <= z <= 1
cvx_end
toc

figure
quiver(psx, psy, cos(ts), sin(ts), 0.2)
hold on
grid on
axis square
plot(ptx, pty, 'ro')
quiver(psx, psy, z.'.*cos(ts), z.'.*sin(ts), 0.2, 'r')

