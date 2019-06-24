
close all
clear
clc

syms ts ptx pty psx psy

t = atan((-sin(ts)*(ptx - psx) + cos(ts)*(pty - psy))/...
         (cos(ts)*(ptx - psx) + sin(ts)*(pty - psy)));

h1 = matlabFunction(diff(t,ptx));
h2 = matlabFunction(diff(t,pty));

ns = 8;

p = (2*pi)/ns*(0:ns-1);

psx = cos(p);
psy = sin(p);
ts = p + pi;
A = zeros(ns,2);
a = zeros(2,2,ns);

for k = 1:ns
    A(k,1) = h1(psx(k), psy(k), 0, 0, ts(k));
    A(k,2) = h2(psx(k), psy(k), 0, 0, ts(k));
    a(:,:,k) = A(k,:).' * A(k,:);
end

a11 = reshape(a(1,1,:),[1 ns]);
a21 = reshape(a(2,1,:),[1 ns]);
a12 = reshape(a(1,2,:),[1 ns]);
a22 = reshape(a(2,2,:),[1 ns]);

k = 4;

cvx_begin
    variable z(ns)
    maximize((a11*z)*(a22*z) + (a21*z)*(a12*z))
    subject to
        ones(1,ns)*z == k
        0 <= z <= 1
cvx_end
