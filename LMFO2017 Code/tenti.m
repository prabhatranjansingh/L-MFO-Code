%TENT MAP BASED CHAOTIC SEQUENCE,ITS 2D MAPPING,3D MAPPING AND
%AUTOCORRELATION IS DEMONSTRATED
% clc;   close all;   clear all;
function k=tenti()
A = .5;  B = 1.99;  phin = .5;  phi(1) = A - (B*phin);

for it = 2:1:6000    
 phi(it) = A - (B*abs(phi(it-1)));    
end
k=phi;
AST = xcorr(phi,phi);
for tt = 1:1:length(phi)-2    
    XX(tt) =  phi(tt+2);
    YY(tt) =  phi(tt+1);
    ZZ(tt) =  phi(tt);    
end    

% figure(1);     plot(phi);            title('\bf TENT MAP');         xlabel('\bf time ');   ylabel('\bf Amplitude');
% figure(2);     plot(ZZ,YY,'k.');     title('\bf Mapping');          xlabel('\bf X(n)');    ylabel('\bf X(n+1)');
% figure(3);     plot(AST);            title('\bf Auto correlation'); xlabel('\bf Time');    ylabel('\bf correlation value');
% figure(4);     plot3(ZZ,YY,XX,'r.'); title('\bf Pseudo phase space trajectories');     grid on;
% xlabel('\bf X(n)');      ylabel('\bf X(n+1)');       zlabel('\bf X(n+2)'); 
%VARY B from 1 to 1.5
