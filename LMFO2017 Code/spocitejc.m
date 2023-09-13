function vektorc=spocitejc(bod)
% vraci vektor c=( - , - , - )

D=length(bod)-11;
bod(1:D+2)=[];
vektorc=[0 0 0];

vetsi_nez_jedna=find(bod>1);
vektorc(1,1)=length(vetsi_nez_jedna);
bod(vetsi_nez_jedna)=0;


vetsi_nez01=find(bod>0.01);
vektorc(1,2)=length(vetsi_nez01);
bod(vetsi_nez01)=0;


vetsi_nez0001=find(bod>0.0001);
vektorc(1,3)=length(vetsi_nez0001);
bod(vetsi_nez0001)=0;


