% function [Best_flame_score,Best_flame_pos,Convergence_curve]=MFO(N,Max_iteration,lb,ub,dim,fobj,num,fmin)
% function [Best_flame_score,Feval,GlobalMins_t12]= MFO(fobj,Max_iteration,Max_FES,N,dim,lb,ub,fmin,varargin)
function [Best_flame_pos,Best_flame_score,Feval]= MFO(fobj,Max_iteration,Max_FES,N,dim,lb,ub,varargin)
num=cell2mat(varargin);
ss=num2str(num);
Function_name=strcat('F',ss);
display('MFO is optimizing your problem');
runtime=30;
Timex=zeros(1,runtime);
tabulka=[];
eps_viol=0.001;
% GlobalMins_t12=zeros(runtime,Max_iteration);
for run=1:runtime
    P=zeros(N,dim+11);
    POM=zeros(N,dim+11);
    succ_rate=0;
    fval=0;
    Feval=0;    
    tic    
    %Initialize the positions of moths
    Moth_pos=initialization(N,dim,ub,lb);
    Convergence_curve=zeros(1,Max_iteration);
    GlobalMins_t=zeros(1,Max_iteration);
    GlobalViol_t=zeros(1,Max_iteration);
    Iteration=1;    
    % Main loop
    while Iteration<Max_iteration+1        
        % Number of flames Eq. (3.14) in the paper
        Flame_no=round(N-Iteration*((N-1)/Max_iteration));        
        for i=1:size(Moth_pos,1)
            % Check if moths go out of the search spaceand bring it back
            Flag4ub=Moth_pos(i,:)>ub;
            Flag4lb=Moth_pos(i,:)<lb;
            Moth_pos(i,:)=(Moth_pos(i,:).*(~(Flag4ub+Flag4lb)))+ub.*Flag4ub+lb.*Flag4lb;
            P(i,1:dim)=Moth_pos(i,:);
            % Calculate the fitness of moths
            [P(i,dim+1),gf,hf]=feval(fobj,Moth_pos(i,:),varargin{:});
            [newviol(i),newg_res(i,:),newh_res(i,:)]=violation_velke(varargin{:},gf,hf,eps_viol,i);
            fval=fval+1;
            P(i,dim+2)=newviol(i);
            P(i,dim+3:dim+5)=newg_res(i,:);
            P(i,dim+6:dim+11)=newh_res(i,:);
        end
        POM=P;        
        if Iteration==1
            % Sort the first population of moths
            POM=sortrows(POM,dim+2);
            POM=sortrows(POM,dim+1);
%             [fitness_sorted I]=sort(Moth_fitness);
%             sorted_population=Moth_pos(I,:); 
            sorted_population=POM;
            % Update the flames
            best_flames=sorted_population;
            best_flame_fitness=POM(:,dim+1);
        else            
            % Sort the moths
            double_population=[previous_population;best_flames];
%             double_fitness=[previous_fitness best_flame_fitness];
            POM=double_population;
            POM=sortrows(POM,dim+2);
            POM=sortrows(POM,dim+1);
            
%             [double_fitness_sorted I]=sort(double_fitness);
%             double_sorted_population=double_population(I,:);
            
%             fitness_sorted=double_fitness_sorted(1:N);
            sorted_population=POM(1:N,:);
            
%             fitness_sorted=double_fitness_sorted(1:N);
%             sorted_population=double_sorted_population(1:N,:);
            
            % Update the flames
            best_flames=POM(1:N,:);
            best_flame_fitness=POM(:,dim+1);
        end
        % Update the position best flame obtained so far
        %         Best_flame_score=fitness_sorted(1);
        %         Best_flame_pos=sorted_population(1,:);
        %         previous_population=Moth_pos;
        %         previous_fitness=Moth_fitness;
        Best_flame_score=POM(1,dim+1);
        Best_flame_pos=POM(1,:);
        previous_population=P;
        previous_fitness=P(:,dim+1);
        % a linearly dicreases from -1 to -2 to calculate t in Eq. (3.12)
        a=-1+Iteration*((-1)/Max_iteration);
        for i=1:size(Moth_pos,1)       
            for j=1:size(Moth_pos,2)
                if i<=Flame_no % Update the position of the moth with respect to its corresponsing flame                    
                    % D in Eq. (3.13)
                    distance_to_flame=abs(sorted_population(i,j)-P(i,j));
                    b=1;
                    t=(a-1)*rand+1;                    
                    % Eq. (3.12)
                    Moth_pos(i,j)=distance_to_flame*exp(b.*t).*cos(t.*2*pi)+sorted_population(i,j);
                end                
                if i>Flame_no % Upaate the position of the moth with respct to one flame                    
                    % Eq. (3.13)
                    distance_to_flame=abs(sorted_population(i,j)-P(i,j));
                    b=1;
                    t=(a-1)*rand+1;                    
                    % Eq. (3.12)
                    Moth_pos(i,j)=distance_to_flame*exp(b.*t).*cos(t.*2*pi)+sorted_population(Flame_no,j);
                end                
            end            
        end        
%         Convergence_curve(Iteration)=Best_flame_score;        
        % Display the iteration and best optimum obtained so far
        if mod(Iteration,50)==0
            display(['At iteration ', num2str(Iteration), ' the best fitness is ', num2str(Best_flame_score)]);
        end
        GlobalMins_t(1,Iteration)=Best_flame_score;
        GlobalViol_t(1,Iteration)=POM(1,dim+2);
        Divv(1,Iteration)=(1/N)*(sum(sqrt(sum((Moth_pos-ones(N,1)*mean(Moth_pos)).^2))));
        Iteration=Iteration+1;
    end
%     violmin=Best_flame_pos(1,d+2);
%     fmin=Best_flame_pos(1,d+1);
    v1=POM(1,dim+2);
    f1=POM(1,dim+1);
    [violmin, indexviolmin]=min(P(:,dim+2)); %minimalizuju violation
    pomocna=P(indexviolmin,:);
    [fmin,indexfmin]=min(pomocna(:,dim+1));
    bodmin=pomocna(indexfmin,:);
%     viol_fun(r,:)=Best_flame_pos(1,d+2);
%     violFun_min(r,:)=Best_flame_pos(1,d+1);
    
    [fmin1,indexfmin1]=min(P(:,dim+1));                                       % based on function
    [violmin1, indexviolmin1]=min(P(indexfmin1,dim+2));
    fun_viol(run,:)=violmin1;
    funViol_min(run,:)=fmin1;

%     c1=spocitejc(bodmin1);
%     c2=spocitejc(bodmin2);
    c=spocitejc(bodmin);
%     por1=length(find(bodmin1(D+3:D+11)>0));
%     por2=length(find(bodmin2(D+3:D+11)>0));
    por=length(find(bodmin(dim+3:dim+11)>0));    
    viol_fun(run,:)=violmin;
    violFun_min(run,:)=fmin;
    fprintf('\n fun=%d run=%d iter=%d ObjVal=%g violmin=%g\n',varargin{:},run,Iteration,fmin,violmin);
    Timex(1,run)=toc;
    vystup=[num run Iteration violmin fmin violmin1 fmin1 c por fval Timex(1,run)];
    tabulka=[tabulka;vystup];
%     Succ(r)=succ_rate;
    

    
%     Diversity(run,:)= Divv;
%     runn(run)=run;
%     Global_min(num,run)=Best_flame_score;
%     GlobalMins(run)=Best_flame_score;
%     minimum=min(GlobalMins_t);
%     StandarDev(run)=std(GlobalMins_t);
%     Mean_run(run)=mean(GlobalMins_t);
    GlobalMins_t12(run,:)=GlobalMins_t;
    Globalviol_t12(run,:)=GlobalViol_t;

    
    
    % plot(1:200,Convergence_curve(1:200))
    % % time(run)=toc;
    % % GlobalMins(run)=Best_flame_score;
    % % Global_min(num,run)=Best_flame_score;
    % % GlobalMins_t12(run,:)=GlobalMins_t;
    % % Run(run)=run;
    % % TFE(run)=Feval;
    % % Feval1(run)=fval;
    % % fmin1(run)=fmin;
    % % Success_run(run)=succ_rate;
    % %     if Feval>=1
    % %         Feval2(run)=Feval;
    % %     else
    % %         Feval2(run)=fval;
    % %     end
end

xlswrite(strcat('C:\Users\Prabhat\Desktop\CEC2017\MFO\MFO_@D',num2str(dim),'pop',num2str(N),'Run',num2str(runtime),'.xlsx'),[tabulka ], [Function_name 'fitRI']);
xlswrite(strcat('C:\Users\Prabhat\Desktop\CEC2017\MFO\MFO_@D',num2str(dim),'pop',num2str(N),'Final.xlsx'),[num mean(tabulka(:,5)) std(tabulka(:,5)) min(tabulka(:,5)) max(tabulka(:,5)) mean(tabulka(:,4)) min(tabulka(:,4)) mean(tabulka(:,7)) std(tabulka(:,7)) min(tabulka(:,7)) max(tabulka(:,7)) mean(tabulka(:,6)) min(tabulka(:,6)) mean(tabulka(:,12)) mean(tabulka(:,13))],Function_name);

xlswrite(strcat('C:\Users\Prabhat\Desktop\CEC2017\MFO\MFO_@D',num2str(dim),'pop',num2str(N),'violFun.xlsx'),[viol_fun],Function_name);
xlswrite(strcat('C:\Users\Prabhat\Desktop\CEC2017\MFO\MFO_@D',num2str(dim),'pop',num2str(N),'violFunMin.xlsx'),[violFun_min],Function_name);

xlswrite(strcat('C:\Users\Prabhat\Desktop\CEC2017\MFO\MFO_@D',num2str(dim),'pop',num2str(N),'funViol.xlsx'),[fun_viol],Function_name);
xlswrite(strcat('C:\Users\Prabhat\Desktop\CEC2017\MFO\MFO_@D',num2str(dim),'pop',num2str(N),'funViolMin.xlsx'),[funViol_min],Function_name);

xlswrite(strcat('C:\Users\Prabhat\Desktop\CEC2017\MFO\MFO_@D',num2str(dim),'pop',num2str(N),'funMin.xlsx'),[GlobalMins_t12],Function_name);
xlswrite(strcat('C:\Users\Prabhat\Desktop\CEC2017\MFO\MFO_@D',num2str(dim),'pop',num2str(N),'ViolMin.xlsx'),[Globalviol_t12],Function_name);


viol_fun=[];
fun_viol=[];
violFun_min=[];
funViol_min=[];
GlobalMins_t12=[];
Globalviol_t12=[];
% Success(num)=sum(Success_run);
%         Time(num)=mean(Timex);
%         TFEval(num)=mean(TFE);      
%         xlswrite('MFO10D30Pop30RunDiversity.xlsx',[Diversity ], [Function_name 'Divr']);
%         xlswrite('MFORun10D30pop30Run.xlsx',[GlobalMins_t12 ], [Function_name 'fitRI']);
%         xlswrite('MFO10D30PopRun.xlsx',[runn' Mean_run' StandarDev' GlobalMins' TFE' Timex'],[Function_name],'A2');
%         xlswrite('MFOFinal10D30pop30Run.xlsx',[num mean(Global_min(num,:)) std(Global_min(num,:)) min(Global_min(num,:)) max(Global_min(num,:)) TFEval(num) Time(num)],Function_name);
save all
end
