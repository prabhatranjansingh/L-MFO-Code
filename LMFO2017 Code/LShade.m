function [FVr_bestmemit,S_bestval,eval]= LShade(fobj,Max_Gen,Max_FES,Particle_Number,Dimension,VRmin,VRmax,varargin)
%Input
D=Dimension;
NP=Particle_Number;
Gmax=Max_Gen;
p=floor(NP*20/100);
%--------------------
num=cell2mat(varargin);
ss=num2str(num);
Function_name=strcat('F',ss);
FVr_bestmem   = zeros(1,D);% best population member ever
runtime=30;
Feval=zeros(1,runtime);

%run-----------------
for run=1:runtime
    tic
    StandarDev=zeros(1,runtime);
    GlobalMins=zeros(1,runtime); 
    Min_run=zeros(1,runtime);
    Feval=zeros(1,runtime);
    %
    eval=0;
    N=NP;
    I_iter=1;    
    A=[];   
    h=1;
    %A=double.empty(0,D);
    FVr_bestmemit = zeros(1,D);% best population member in iteration
    FVr_minbound=ones(1,D).*VRmin;
    FVr_maxbound=ones(1,D).*VRmax;
    %x_val=ones(NP,I_iter);%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    x = zeros(NP,D); %initialize FM_pop to gain speed  
    for i=1:N                          % check the remaining members
      r=randi(h);
      Mcr(r)=0.5;
      Mf(r)=0.5;
      x(i,:) = FVr_minbound + rand(1,D).*(FVr_maxbound - FVr_minbound);
      x_val(i)  = fobj(x(i,:));
      eval=eval+1;
    end
    
    A=[A;x(1,:)];
    while (I_iter < Gmax)
        [P1,I1]=sort(x_val);
        Sf=[];
        Scr=[];
        x_copy=x(:,:);
        N=round(NP-eval*((NP-4)/Max_FES));
        N1=I1([1:N]);
        for i=1:N
            ri=randi(h);            
            if (Mcr(ri)==any(Scr))   %Check Here
                CR(N1(i))=0;
            else
                CR(N1(i))=normrnd(Mcr(ri),0.1); % random number using normal distribution << Use Mcr_ri >>
            end            
            F(N1(i))=Mf(ri)+0.1*trnd(1);   % random number using cauchy distribution with one degree of freedom in t distribution. <<Use Mf_ri>>
%             if ((F(i) < 0) | (F(i) > 1))
%                F(i)=0.5;
%                fprintf(1,'F should be from interval [0,1]; set to default value 0.5\n');
%             end
%              if((CR(i)>1))
%                 CR(i)=1;
%                 fprintf(1,'CR should be from interval [0,1]; set to default value 0.5\n');
%             end
            k=I1(randi([1 p]));
            k1=randi([1,NP]);
            
            k2=union(A,x_copy,'rows');
            [k2d,k2s]=size(k2);
            %k2r=k2(randi([1,k2d]),:);
            v(N1(i),:) = x(N1(i),:) + F(N1(i)).*(x(k,:)-x(N1(i),:)) + F(N1(i)).*(x(k1,:) - k2(randi([1,k2d]),:));
            %Apply inteplolation & crossover method
            u = zeros(NP,D);
            for j=1:D
                if(v(N1(i),j)<FVr_minbound(j))
                    v(N1(i),j)=(FVr_minbound(j)+x(N1(i),j))/2;
                end
                if(v(N1(i),j)>FVr_maxbound(j))
                    v(N1(i),j)=(FVr_maxbound(j)+x(N1(i),j))/2;
                end
%                 v_val(i,I_iter)  = feval(fhd,u(i,:,I_iter),varargin{:});
%                 eval=eval+1;
                %Apply crossover
                if ((rand<=CR(N1(i))) | (j==randi(D)))
                    u(N1(i),j)=v(N1(i),j);
                else
                    u(N1(i),j)=x(N1(i),j);
                end                
            end
            u_val(N1(i))=fobj(u(N1(i),:));%feval(fhd,u(N1(i),:),varargin{:});
            eval=eval+1;
            %Apply selection
            x_copy1=zeros(1,D);
            if(u_val(N1(i))<x_val(N1(i)))
                x_copy1=x(N1(i),:);
                A=[A;x_copy1];
                if(size(A)>NP)
                    A(randi([1,NP]),:)=[];
                end
                Scr=[Scr,CR(N1(i))];
                Sf=[Sf,F(N1(i))];
                
                x(N1(i),:)=u(N1(i),:);
                x_val(N1(i))=u_val(N1(i));
                %fprintf('Changing');
            end
            %
        end
        %[P2,I2]=sort(x_val);
        sz=size(Scr,2);
        w=abs(u_val(1:sz)-x_val(1:sz))./(sum(abs(u_val(1:sz)-x_val(1:sz))));
        
        if(sz~=0)
            if(h>NP)
                h=1;
            else
                h=h+1;
            end
            Mcr(h)=sum(w(1:sz).*Scr(1:sz));
            Mf(h)=sum(w(1:sz).*(Sf(1:sz).^2))/sum(w(1:sz).*Sf(1:sz));
            %h=h+1;
        end
        Divv(1,I_iter)=(1/NP)*(sum(sqrt(sum((x-ones(NP,1)*mean(x)).^2)))); 
        S_bestval=P1(1);
        GlobalMins_t(I_iter)=S_bestval;    
        display(['prob ', num2str(num), ' Run ', num2str(run), 'At iteration ', num2str(I_iter), ' the best fitness is ', num2str(S_bestval)]);           
%        fprintf('%f  Mean:',MeanL);
%        xlswrite('DEJade2.xlsx',[run' I_iter' MeanL' Mf' Mcr' eval'],[Function_name],'A1');
%        fprintf(1,'Problem: %d, Run: %d, Iteration: %d, Mf: %f, Mcr: %f, Best: %f \n',num,run,I_iter,Mf,Mcr,S_bestval);
        FVr_bestmemit=x(I1(1),:);
        I_iter=I_iter+1;
    end    
    GlobalMins_t12(run,:)=GlobalMins_t;
%     Min_run(run)=mean(GlobalMins_t);
%     StandarDev(run)=std(GlobalMins_t);
    GlobalMins(run)=S_bestval;
    Feval(run)=eval;
    fprintf('Problem: %d, Run: %d, Best: %f,  Feval: %d\n',num,run,S_bestval,eval);   
    
    Diversity(run,:)= Divv;
    time(run)=toc;  
    Total(run)=run;
    Global_min(num,run)=S_bestval;
    GlobalMins(run)=S_bestval;     
end

TFEval(num)=mean(Feval);
Time(num)=mean(time);
xlswrite('LShadeDiversityD_5000_N_30_runtime_30.xlsx',[Total' GlobalMins' Feval' time' ],[Function_name],'A2');
xlswrite('LShadeConvergeD_5000_pop_30Run_30.xlsx',[GlobalMins_t12 ], [Function_name 'fitRI']);
xlswrite('LShadeRunD_5000_Pop_30_Run.xlsx',[Diversity ], [Function_name 'Divr']);
xlswrite('LShadeFinalD_5000_pop_30Run.xlsx',[num mean(Global_min(num,:)) std(Global_min(num,:)) min(Global_min(num,:)) max(Global_min(num,:)) TFEval(num) Time(num)],Function_name);

end
