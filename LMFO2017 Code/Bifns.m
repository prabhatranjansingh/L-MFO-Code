%THIS PROGRAM IS WRITTEN FOR DEMONSTRATION OF BIFURCATIONS IN NONLINEAR
%DYNAMICAL SYSTEMS.THIS PARTICULAR DEMONSTRATION IS FOR
%LOGISTIC,QUADRATIC,TENT AND BERNOULLI MAPS.
% clc;   close all;  clear all;
function Fina4=Bifns()
%--------------------------------------------------------------------------
% LOGISTIC MAP
%--------------------------------------------------------------------------
A1 = .5; B1 = 2.9; phin1 = -0.35; Fina = []; 
phi1(1) = ( B1*[(A1^2) - (phin1^2)] ) - A1;

for ii = 1:1:100
    B1 = B1 + .01 ;
for ih = 2:1:300
  phi1(ih) = ( B1*[(A1^2) - (phi1(ih-1)^2)] ) - A1;  
end
Fina = [Fina phi1(20:length(phi1))];
end
%-------------------------------------------------------------------------
% QUADRATIC MAP
%-------------------------------------------------------------------------
A2 = 4; B2 = .1; phin2 = 0.15; Fina1 = [];
phi2(1) = B2 - A2*(phin2^2); 

for ii = 1:1:40
    B2 = B2 + 0.01;
for ib = 2:1:600    %%400
phi2(ib) = B2- (A2*(phi2(ib-1).^2));    
end
Fina1 = [Fina1 phi2(7:length(phi2))];
end
%--------------------------------------------------------------------------
%TENT MAP
%--------------------------------------------------------------------------
A3 = .5; B3 = .6; phin3 = .5; Fina3 = [];
phi3(1) = A3 - (B3*phin3);

for ii = 1:1:90
    B3 = B3 + 0.01;   
for it = 2:1:300    
 phi3(it) = A3 - (B3*abs(phi3(it-1)));    
end
Fina3 = [Fina3 phi3(40:length(phi3))];
end
%--------------------------------------------------------------------------
%BERNOULLI MAP
%--------------------------------------------------------------------------
B = 0.0;  A = .5;  phin = 0.25;  phi(1) = (B*phin) - A; Fina4 = [];
for gg = 1:1:199
     B = B + 0.01;
for ii = 2:1:60        
if phi(ii-1) > 0        
phi(ii) = (B*phi(ii-1)) - A;
else      
phi(ii) = (B*phi(ii-1)) + A;        
end
end
Fina4 = [Fina4 phi(10:length(phi))];
end

% figure(1);     plot(Fina,'r.');      title('\bf Bifurcation plot of LOGISTIC map');
% figure(2);     plot(Fina1,'r.');     title('\bf Bifurcation plot of QUADRATIC map');
% figure(3);     plot(Fina3,'r.');     title('\bf Bifurcation plot of TENT MAP');
% figure(4);     plot(Fina4,'r.');     title('\bf Bifurcation plot of BERNOULLI MAP');

