function out1 = h2(psx,psy,ptx,pty,ts)
%H2
%    OUT1 = H2(PSX,PSY,PTX,PTY,TS)

%    This function was generated by the Symbolic Math Toolbox version 7.2.
%    24-Jun-2019 09:59:39

t2 = cos(ts);
t3 = psy-pty;
t4 = sin(ts);
t5 = psx-ptx;
t11 = t2.*t3;
t12 = t4.*t5;
t6 = t11-t12;
t7 = t2.*t5;
t8 = t3.*t4;
t9 = t7+t8;
t10 = 1.0./t9.^2;
out1 = -(t2./t9-t4.*t6.*t10)./(t6.^2.*t10+1.0);
