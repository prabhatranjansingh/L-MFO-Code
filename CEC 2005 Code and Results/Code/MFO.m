% function [Best_flame_score,Best_flame_pos,Convergence_curve]=MFO(N,Max_iteration,lb,ub,dim,fobj,num,fmin)
% function [Best_flame_score,Feval,GlobalMins_t12]= MFO(fobj,Max_iteration,Max_FES,N,dim,lb,ub,fmin,varargin)
function [Best_flame_pos,Best_flame_score,Feval]= MFO(fobj,Max_iteration,Max_FES,N,dim,lb,ub,varargin)
num=cell2mat(varargin);
ss=num2str(num);
Function_name=strcat('F',ss);
display('MFO is optimizing your problem');
runtime=30;
% GlobalMins_t12=zeros(runtime,Max_iteration);
for run=1:runtime
    tic 
      succ_rate=0;
      fval=0;
      Feval=0;
%Initialize the positions of moths
Moth_pos=initialization(N,dim,ub,lb);

Convergence_curve=zeros(1,Max_iteration);
GlobalMins_t=zeros(1,Max_iteration);
Iteration=1;
record=zeros(Max_iteration,N);
% Main loop
poprecord=[];
while Iteration<Max_iteration+1
    
    % Number of flames Eq. (3.14) in the paper
    Flame_no=round(N-Iteration*((N-1)/Max_iteration));
    
    for i=1:size(Moth_pos,1)
        
        % Check if moths go out of the search spaceand bring it back
        Flag4ub=Moth_pos(i,:)>ub;
        Flag4lb=Moth_pos(i,:)<lb;
        Moth_pos(i,:)=(Moth_pos(i,:).*(~(Flag4ub+Flag4lb)))+ub.*Flag4ub+lb.*Flag4lb;    
        Moth_fitness(1,i)=fobj(Moth_pos(i,:));
        fval=fval+1;
    end   
    if Iteration==1
        % Sort the first population of moths
        [fitness_sorted I]=sort(Moth_fitness);
        sorted_population=Moth_pos(I,:);        
        % Update the flames
        best_flames=sorted_population;
        best_flame_fitness=fitness_sorted;
    else
        
        % Sort the moths
        double_population=[previous_population;best_flames];
        double_fitness=[previous_fitness best_flame_fitness];
        
        [double_fitness_sorted I]=sort(double_fitness);
        double_sorted_population=double_population(I,:);
        
        fitness_sorted=double_fitness_sorted(1:N);
        sorted_population=double_sorted_population(1:N,:);
        
        % Update the flames
        best_flames=sorted_population;
        best_flame_fitness=fitness_sorted;
    end
    record(Iteration,:)=fitness_sorted;
    moth_record(:,:)=sorted_population;
    
    % Update the position best flame obtained so far
    Best_flame_score=fitness_sorted(1);
    Best_flame_pos=sorted_population(1,:);
      
    previous_population=Moth_pos;
    previous_fitness=Moth_fitness;
    
    % a linearly dicreases from -1 to -2 to calculate t in Eq. (3.12)
    a=-1+Iteration*((-1)/Max_iteration);
    
    for i=1:size(Moth_pos,1)
        
        for j=1:size(Moth_pos,2)
            if i<=Flame_no % Update the position of the moth with respect to its corresponsing flame                
                % D in Eq. (3.13)
                distance_to_flame=abs(sorted_population(i,j)-Moth_pos(i,j));
                b=1;
                t=(a-1)*rand+1;                
                % Eq. (3.12)
                Moth_pos(i,j)=distance_to_flame*exp(b.*t).*cos(t.*2*pi)+sorted_population(i,j);
            end            
            if i>Flame_no % Upaate the position of the moth with respct to one flame                
                % Eq. (3.13)
                distance_to_flame=abs(sorted_population(i,j)-Moth_pos(i,j));    % update only correspond to Flame_no sorted_population(Flame_no,j)
                b=1;
                t=(a-1)*rand+1;                
                % Eq. (3.12)
                Moth_pos(i,j)=distance_to_flame*exp(b.*t).*cos(t.*2*pi)+sorted_population(Flame_no,j);
            end
            
        end
        poprecord(i,Iteration)=Moth_fitness(1,i);
    end
    Convergence_curve(Iteration)=Best_flame_score;
    
    % Display the iteration and best optimum obtained so far
    %if mod(Iteration,100)==0
        %display(['At iteration ', num2str(Iteration), ' the best fitness is ', num2str(Best_flame_score)]);
        display(['Function ',ss,'At Run ',num2str(run),'iteration ', num2str(Iteration), ' the best fitness is ', num2str(Best_flame_score)]);
    %end
    GlobalMins_t(1,Iteration)=Best_flame_score;
    Divv(1,Iteration)=(1/N)*(sum(sqrt(sum((Moth_pos-ones(N,1)*mean(Moth_pos)).^2))));
    Iteration=Iteration+1; 
end
boxplot(poprecord,'Plotstyle','compact');
%sheet1='Record'
%xlswrite('Record.xlsx',[record],sheet1);

records(run,:,:)=record;
Diversity(run,:)= Divv;
runn(run)=run;
Global_min(num,run)=Best_flame_score;
GlobalMins(run)=Best_flame_score;
minimum=min(GlobalMins_t);
StandarDev(run)=std(GlobalMins_t);
Mean_run(run)=mean(GlobalMins_t);
GlobalMins_t12(run,:)=GlobalMins_t;
TFE(run)=fval;
Timex(run)=toc;
%%plot(1:200,Convergence_curve(1:200))
time(run)=toc;
GlobalMins(run)=Best_flame_score;
Global_min(num,run)=Best_flame_score;
GlobalMins_t12(run,:)=GlobalMins_t;
Run(run)=run;
TFE(run)=Feval;
Feval1(run)=fval;
% % fmin1(run)=fmin;
Success_run(run)=succ_rate;
if Feval>=1
    Feval2(run)=Feval;
else
    Feval2(run)=fval;
end
end
Success(num)=sum(Success_run);
Time(num)=mean(Timex);
TFEval(num)=mean(TFE);      
% xlswrite('MFO_O_DiversityD_2000_N_30_runtime_30.xlsx',[Diversity ], [Function_name 'Divr']);
% xlswrite('MFO_O_ConvergeD_2000_pop_30Run_30.xlsx',[GlobalMins_t12 ], [Function_name 'fitRI']);
% xlswrite('MFO_O_RunD_2000_Pop_30_Run.xlsx',[runn' GlobalMins' TFE' Timex'],[Function_name],'A2');
% xlswrite('MFO_O_FinalD_2000_pop_30Run.xlsx',[num mean(Global_min(num,:)) std(Global_min(num,:)) min(Global_min(num,:)) max(Global_min(num,:)) TFEval(num) Time(num)],Function_name);
save all
end
