function [gbest,GlobalMin,Feval]= fitSHS(fhd,MaxIt1,Max_FES,nPop,nVar,VarMin,VarMax,varargin)
    ss=cell2mat(varargin);
    ss1=num2str(ss);
    Function_name=strcat('F',ss1);
    VarSize=[1 nVar];   % Size of Decision Variables Matrix
    runtime=1;
    Feval=0;
    Timex=zeros(1,runtime);
    for r=1:runtime
        tic
        MaxIt = MaxIt1;     % Maximum Number of Iterations

        HMS = 15;         % Harmony Memory Size

        nNew = 15;        % Number of New Harmonies

                
        
        %IHS
        PARmin = 0.1;        % Pitch Adjustment Rate
        PARmax=0.99;        

        FWmax = 0.02*(VarMax-VarMin);    % Fret Width (Bandwidth)
        FWmin=0.0001;

        %% Initialization

        for i = 1:HMS
            HM(i,:) = unifrnd(VarMin, VarMax, VarSize);
            HMval(i)=feval(fhd,HM(i,:),varargin{:});
            Feval=Feval+1;
        end
        [~, SortOrder] = sort(HMval);
        HM = HM(SortOrder,:);

        % Update Best Solution Ever Found
% % %         BestSol = HM(1);
        BestSol = HMval(1);
        par_val = zeros(MaxIt, 1);
        fw_val = zeros(MaxIt, 1);
        % Array to Hold Best Cost Values
        BestCost = zeros(MaxIt, 1);
        % Initialize Array for New Harmonies
%         NEW = repmat([], nNew, 1);        
        
        NEW=HM;
        NEWval=HMval;
        Varmin=ones(1,nVar)*VarMin;
        Varmax=ones(1,nVar)*VarMax;
        fit1=4;
        fit2=0;
        fit3=4;

        %% Harmony Search Main Loop
        for it = 1:MaxIt
            A=[];
            if(fit1==0)
                select=1;
            elseif(fit2==0)
                select=2;
            elseif(fit3==0)
                select=3;
            end
            
            if (select==1)
                HMCR = 0.9;       % Harmony Memory Consideration Rate
                PAR=PARmin+((PARmax-PARmin)/MaxIt)*it;                      %IHS
                FW = FWmax*exp((log(FWmin/FWmax)/MaxIt)*it);                
            elseif(select==2)
                HMCR = 0.9;       % Harmony Memory Consideration Rate
                PAR = 0.6;        % Pitch Adjustment Rate                   %Original HS
                FW = 0.02*(VarMax-VarMin);    % Fret Width (Bandwidth)
                FW_damp = 0.997;              % Fret Width Damp Ratio
                % Damp Fret Width
                FW = FW*FW_damp;
            else
                for j=1:nVar
                    Varmin(j)=max(min(HM(:,j))-FWmax,VarMin);
                    Varmax(j)=min(max(HM(:,j))-FWmax,VarMax);
                end
                PAR=PARmin+((PARmax-PARmin)/MaxIt)*it;                      %GDHS
                FW = FWmax*exp((log(FWmin/FWmax)/MaxIt)*it);
                HMCR=0.9+0.2*sqrt(((it-1)/(MaxIt-1))*(1-((it-1)/(MaxIt-1))));
            end
%             A=[A HMval];
            % Create New Harmonies
            for k = 1:nNew
                % Create New Harmony Position
                for j = 1:nVar
                    if rand <= HMCR
                        % Use Harmony Memory
                        i = randi([1 HMS]);
                        NEW(k,j) = HM(i,j);
                        % Pitch Adjustment
                        if rand <= PAR
                            DELTA = FW*rand();            % Gaussian (Normal) 
                            if rand<=0.5
                                NEW(k,j)= NEW(k,j)+DELTA;
                            else                                
                                NEW(k,j)= NEW(k,j)-DELTA;
                            end
                        end
                    else                        
                        %if(select==3)
                        NEW(k,j) = Varmin(j)+rand*(Varmax(j)-Varmin(j));
%                         else
%                             NEW(k,j) = VarMin+rand*(VarMax-VarMin);
%                         end
                    end
                end
                % Evaluation
                NEWval(k)=feval(fhd,NEW(k,:),varargin{:});
                Feval=Feval+1;
                [val2,ind2]=max([HMval]);
% % %                 if(NEWval<val2)                                    %Worst update
% % %                     HMval(ind2)=NEWval(k);
% % %                     HM(ind2,:)=NEW(k,:);                                  
% % %                 end   
                if(select==1)
                    if(NEWval(k)<val2)
                        HMval(ind2)=NEWval(k);
                        HM(ind2,:)=NEW(k,:);
                        fit1=0;
                    else
                        fit1=4;
                        fit2=0;
                    end
                elseif(select==2)
                    if(NEWval(k)<val2)
                        HMval(ind2)=NEWval(k);
                        HM(ind2,:)=NEW(k,:);
                        fit2=0;
                    else
                        fit2=4;
                        fit3=0;
                    end
                else
                    if(NEWval(k)<val2)
                        HMval(ind2)=NEWval(k);
                        HM(ind2,:)=NEW(k,:);
                        fit3=0;
                    else
                        fit3=4;
                        fit1=0;
                    end
                end
            end             
            [~, SortOrder] = sort([HMval]);
            HM = HM(SortOrder,:);            
            
            % Update Best Solution Ever Found
            BestSol = HMval(1);
            BestPos=HM(1,:);
            % Store Best Cost Ever Found
            BestCost(it) = BestSol;

            % Show Iteration Information
            disp(['Function' num2str(ss) '  Run' num2str(r) 'Iteration ' num2str(it) ': Best Cost = ' num2str(BestSol)]);
            GlobalMins_t(1,it)=BestSol;
            gbest=BestSol;
            par_val(it)=PAR;
            fw_val(it)=FW;
        end

        %% Results

        figure;
        % plot(BestCost, 'LineWidth', 2);
        semilogy(BestCost, 'LineWidth', 2);
        xlabel('Iteration');
        ylabel('Best Cost');
        grid on;
        
        figure;
        plot(fw_val, 'LineWidth', 2);
%         semilogy(fw_val, 'LineWidth', 2);
        xlabel('Iteration');
        ylabel('fw_val');
        grid on;
        
        run(r)=r;
        Global_min(ss,r)=GlobalMins_t(end);
        GlobalMins(r)=GlobalMins_t(end);
        GlobalMins_t12(r,:)=GlobalMins_t;
        TFE(r)=Feval;
        Timex(r)=toc;
    end
    GlobalMin=min(GlobalMins);
    Time(ss)=mean(Timex);
    TFEval(ss)=mean(TFE);        
    xlswrite('HSRun10D30pop30Run.xlsx',[GlobalMins_t12 ], [Function_name 'fitRI']);
    xlswrite('HS10D30PopRun.xlsx',[run' GlobalMins' TFE' Timex'],[Function_name],'A2');
    xlswrite('HSFinal10D30pop30Run.xlsx',[ss mean(Global_min(ss,:)) std(Global_min(ss,:)) min(Global_min(ss,:)) max(Global_min(ss,:)) TFEval(ss) Time(ss)],Function_name);
    save all
end

