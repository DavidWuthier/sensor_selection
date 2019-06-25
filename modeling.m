%%

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

% # of sensor
m = 40;
% # of target points
np = 20;
% Time
t = 0:(2*pi)/np:2*pi*(1-1/np);
% # target points
ptx = 0.3*cos(t);
pty = 0.3*sin(2*t);
% Azimut of cameras
p = (2*pi)/m*(0:m-1);
% Camera positions
psx = cos(p);
psy = sin(p);
% Camera angles
ts = p + pi;

% Jacobian of the measurements
A = zeros(np,2*np,m);
for i = 1:m
    for j = 1:np
        A(j,2*j-1,i) = h1(psx(i), psy(i), ptx(j), pty(j), ts(i));
        A(j,2*j,i) = h2(psx(i), psy(i), ptx(j), pty(j), ts(i));
    end
end

k = 4;
h = 0.01 *2/m;

cvx_begin
    variable z(m)
    expressions s1(2*np,2*np,m) s2(m)
    
    for i = 1:m
        s1(:,:,i) = z(i) * A(:,:,i).' * A(:,:,i);
        s2(i) = log(z(i)) + log(1 - z(i));
    end
    
    maximize(log_det(sum(s1, 3)) + h*sum(s2))
    
    subject to
        ones(1,m)*z == k
cvx_end

for i = 1:m
    s1(:,:,i) = z(i) * A(:,:,i).' * A(:,:,i);
end

ps = log(det(sum(s1, 3)))
pwc = ps + 2*m*h

zs = sort(z);
zl = z >= zs(end-k+1);

for i = 1:m
    s1(:,:,i) = zl(i) * A(:,:,i).' * A(:,:,i);
end

psl = log(det(sum(s1, 3)))

e = (psl - pwc)/pwc

%%

s = 0.5;
c = [0 0.4470 0.7410];

figure
hold on
grid on
axis square
quiver(psx, psy, s*cos(ts), s*sin(ts), 'Color', [0.7 0.7 0.7], 'LineWidth', 2, 'AutoScale', 'off', 'MaxHeadSize', 0.1)
plot(ptx, pty, 'ro', 'LineWidth', 2)
quiver(psx(zl), psy(zl), s*cos(ts(zl)), s*sin(ts(zl)), 'g', 'LineWidth', 2, 'AutoScale', 'off', 'MaxHeadSize', 0.2)
quiver(psx, psy, s*z.'.*cos(ts), s*z.'.*sin(ts), 'b', 'LineWidth', 2, 'ShowArrowHead', 'off', 'AutoScale', 'off')
xlabel('p_x [m]')
ylabel('p_y [m]')
legend('locations', 'target', 'selected', 'z')


