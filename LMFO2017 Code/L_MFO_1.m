function [Best_flame_pos,Best_flame_score,Feval]= L_MFO_1(fobj,Max_iteration,Max_FES,N,dim,lb,ub,varargin)
num=cell2mat(varargin);
ss=num2str(num);
Function_name=strcat('F',ss);
display('MFO is optimizing your problem');
runtime=30;
UB=ones(1,dim).*ub;
LB=ones(1,dim).*lb;
tabulka=[];
eps_viol=0.001;
for run=1:runtime
    P=zeros(N,dim+11);
    L=zeros(1,dim+11);
    POM=zeros(N,dim+11);
    succ_rate=0;
    fval=0;
    Feval=0;
    tic 
    %Initialize the positions of moths
    Moth_pos=initialization(N,dim,ub,lb);
    L_pos=initialization(1,dim,ub,lb);
    Convergence_curve=zeros(1,Max_iteration);
    GlobalMins_t=zeros(1,Max_iteration);
    GlobalViol_t=zeros(1,Max_iteration);
    Iteration=1;
    record=zeros(Max_iteration,N);
    %%%%%%%%%%%%%%%%%%%% Let we take the value of minpts and maxpts as user input
    minpts=3;
    maxpts=size(Moth_pos,1);
    Toutliers=[];
    fw=1;
    Flag4ub=L_pos(1,:)>ub;
    Flag4lb=L_pos(1,:)<lb;
    L_pos(1,:)=(L_pos(1,:).*(~(Flag4ub+Flag4lb)))+ub.*Flag4ub+lb.*Flag4lb;
    L(1,1:dim)=L_pos(1,:);
    [L(1,dim+1),Lgf,Lhf]=feval(fobj,L_pos(1,:),varargin{:});
    [Lviol(1),Lg_res(1,:),Lh_res(1,:)]=violation_velke(varargin{:},Lgf,Lhf,eps_viol,1);
    L(1,dim+2)=Lviol(1);
    L(1,dim+3:dim+5)=Lg_res(1,:);
    L(1,dim+6:dim+11)=Lh_res(1,:);
    for i=1:size(Moth_pos,1)
        % Check if moths go out of the search spaceand bring it back
        Flag4ub=Moth_pos(i,:)>ub;
        Flag4lb=Moth_pos(i,:)<lb;
        Moth_pos(i,:)=(Moth_pos(i,:).*(~(Flag4ub+Flag4lb)))+ub.*Flag4ub+lb.*Flag4lb;
        % Calculate the fitness of moths\
        P(i,1:dim)=Moth_pos(i,:);
        % Calculate the fitness of moths
        [P(i,dim+1),gf,hf]=feval(fobj,Moth_pos(i,:),varargin{:});
        [newviol(i),newg_res(i,:),newh_res(i,:)]=violation_velke(varargin{:},gf,hf,eps_viol,i);
        fval=fval+1;
        P(i,dim+2)=newviol(i);
        P(i,dim+3:dim+5)=newg_res(i,:);
        P(i,dim+6:dim+11)=newh_res(i,:);
        fval=fval+1;
    end
    % Main loop
    while Iteration<Max_iteration+1
        % Number of flames Eq. (3.14) in the paper
        Flame_no=round(N-Iteration*((N-1)/Max_iteration));        
        % previous_population=P;                                               % this and Lizard
%         POM=P;
        if Iteration==1
            % Sort the first population of moths
            POM=sortrows(P,dim+2);
            POM=sortrows(POM,dim+1);
            sorted_population=POM;
            % Update the flames
            best_flames=sorted_population;
            best_flame_fitness=POM(:,dim+1);
        else
            % Sort the moths
            double_population=[previous_population;best_flames];
%             POM=double_population;
            double_population=sortrows(double_population,dim+2);
            double_population=sortrows(double_population,dim+1);
            sorted_population=double_population(1:N,:);
            POM=sorted_population;
            % Update the flames
            best_flames=sorted_population;
            best_flame_fitness=sorted_population(:,dim+1);
        end
        previous_population=P; 
        % Update the position best flame obtained so far
        Best_flame_score=POM(1,dim+1);
        Best_flame_pos=POM(1,:);
        a=-1+Iteration*((-1)/Max_iteration);                               % a linearly dicreases from (-1 to -2) to calculate t in Eq. (3.12)        
        %% Moth position
        Index2=[1:size(Moth_pos,1)]';                                     % 1,2,3,..... 30(Moth size)
        %%Now we apply the DBSCAN Clustering algroithm
        %%%%%%%%%%%%%%%%%%%%find the epsilon value%%%%%%%%%%%%%%%%%%%%%%%%%%
        epsilon2=clusterDBSCAN.estimateEpsilon(P(:,1:dim),minpts,maxpts);        
        %%%%%%%%%%%%%%%%%%%After getting the value of epsilon we impelemt the
        %%%%%%%%%%%%%%%%%%%dbscan alogrithm%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        cluster_label=dbscan(P(:,1:dim),epsilon2,minpts);                    % cluster_label is idx2 
        unique_idx2=unique(cluster_label);
        %here we add the labels into data frame
        T2=table(Index2,P(:,1:dim),cluster_label);
        %%%Now we find each cluster.%%%
        Cluster_Labels2=[];             %n21
        Size_of_each_cluster2=[];       %n22
        for i=1:height(unique_idx2)
            Cluster_Labels2=[Cluster_Labels2;unique_idx2(i)];        
            %size of each cluster
            Size_of_each_cluster2=[Size_of_each_cluster2;height(T2(T2.cluster_label==unique_idx2(i),:))]; % Seperate with cluster level
        end       
        dataframe2=table(Cluster_Labels2,Size_of_each_cluster2);           % Determination of all clusters and its size finished.
        %%%%%%%%%%%For Outlier Work%%%%%%%%%%%%%%
        if any(dataframe2.Cluster_Labels2==-1)
            %(i) Find the indexes of Outliers and store into the excel
            for i=1:length(unique_idx2)
                if unique_idx2(i)==-1
                    Toutliers=T2(T2.cluster_label==-1,:);
                end
            end
        end
        %%%%%%%%%%%%%%%%%%%%%For Non Outlier Work%%%%%%%%%%%%%%%%%%%%%%
        Noutliers=dataframe2(dataframe2.Cluster_Labels2~=-1,:);
        %%%%%%%%%%%%maximum and minimum cluster find%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        if height(Noutliers)~=0
            %%%%%%%%%%%%%%%%%%%%%%%minimum size cluster%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            minimum_table2=Noutliers(Noutliers.Size_of_each_cluster2==min(Noutliers.Size_of_each_cluster2),:);
            %%here we check that wheather more than one cluster comes under minimum cluster tag or not
            if height(minimum_table2)>=1
                minimum_table2=head(minimum_table2,1);
                %Here we store the index of value which belong into that particular cluster
                for i=1:length(unique_idx2)
                    if unique_idx2(i)==minimum_table2.Cluster_Labels2
                        MinCluster=T2(T2.cluster_label==minimum_table2.Cluster_Labels2,:);                 
                    end
                end
            end
            %%%%%%%%%%%%%Maximum size cluster%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            maximum_table2=Noutliers(Noutliers.Size_of_each_cluster2==max(Noutliers.Size_of_each_cluster2),:);
            %%here we check that wheather more than one cluster comes under maximum cluster tag or not
            if height(maximum_table2)>=1
                maximum_table2=head(maximum_table2,1);
                %(ii) Here we store the index of value which belong into that particular cluster
                for i=1:length(unique_idx2)
                    if unique_idx2(i)==maximum_table2.Cluster_Labels2
                        MaxCluster=T2(T2.cluster_label==maximum_table2.Cluster_Labels2,:);
                    end
                end
            end
            %%%%%%%%%%%%%%%%%%%%%Model Prepared%%%%%%%%%%%%%%%%%%%%%%%
    %% ------------------------------------------------------------------------
            %Randomly select individual from (Heighest size) cluster
            k=randperm(height(MaxCluster));      %MaxCluster
            random_val=MaxCluster(k(1),:);       %MaxCluster
            %%%update the index of the outliers towards the maximum cluster
            if height(Toutliers)~=0
                for i=1:height(Toutliers)
                    %% update outliers towards the maximum cluster
                    P(Toutliers.Index2(i),1:dim)=rand.*P(Toutliers.Index2(i),1:dim)+(rand.*P(Toutliers.Index2(i),1:dim)-P(Toutliers.Index2(i),1:dim)).*(-1+2*rand)+...
                        rand.*(P(random_val.Index2,1:dim)-P(Toutliers.Index2(i),1:dim));
                    [P(Toutliers.Index2(i),dim+1),gf,hf]=feval(fobj,P(Toutliers.Index2(i),1:dim),varargin{:});
                    [newviol(i),newg_res(i,:),newh_res(i,:)]=violation_velke(varargin{:},gf,hf,eps_viol,i);
                    P(i,dim+2)=newviol(i);
                    P(i,dim+3:dim+5)=newg_res(i,:);
                    P(i,dim+6:dim+11)=newh_res(i,:);
                     %%%%%%%%%   
                    % K_P(1,1:dim)=rand.*P(Toutliers.Index2(i),1:dim)+(rand.*P(Toutliers.Index2(i),1:dim)-P(Toutliers.Index2(i),1:dim)).*(-1+2*rand)+...
                    %         rand.*(P(random_val.Index2,1:dim)-P(Toutliers.Index2(i),1:dim));
                    % K_Q(1,1:dim)=(UB+LB-P(Toutliers.Index2(i),1:dim));
                    % 
                    % [K_P(i,dim+1),KPgf,KPhf]=feval(fobj,K_P(1,1:dim),varargin{:});
                    % [K_Q(i,dim+1),KQgf,KQhf]=feval(fobj,K_Q(1,1:dim),varargin{:});
                    % [K_Pviol(i),KPg_res(i,:),KPh_res(i,:)]=violation_velke(varargin{:},KPgf,KPhf,eps_viol,i);
                    % [K_Qviol(i),KQg_res(i,:),KQh_res(i,:)]=violation_velke(varargin{:},KQgf,KQhf,eps_viol,i);
                    % 
                    % 
                    % K_P(i,dim+2)=K_Pviol(i);
                    % K_P(i,dim+3:dim+5)=KPg_res(i,:);
                    % K_P(i,dim+6:dim+11)=KPh_res(i,:);
                    % K_Q(i,dim+2)=K_Qviol(i);
                    % K_Q(i,dim+3:dim+5)=KQg_res(i,:);
                    % K_Q(i,dim+6:dim+11)=KQh_res(i,:);
                    % 
                    % 
                    % if((K_P(i,dim+2)==0) && (K_Q(i,dim+2)==0))                      % Check here ones completed
                    %     if(K_P(i,dim+1)<K_Q(i,dim+1))
                    %         P(Toutliers.Index2(i),:)= K_P(i,:);
                    %     else
                    %         P(Toutliers.Index2(i),:)= K_Q(i,:);
                    %     end
                    % elseif(K_P(i,dim+2)<K_Q(i,dim+2))
                    %     P(Toutliers.Index2(i),:)= K_P(i,:);
                    % else
                    %     P(Toutliers.Index2(i),:)= K_Q(i,:);
                    % end                
                    fval=fval+1;
                end
            end
            %Randomly select individual from (Lowest size) cluster
            if height(Noutliers)~=1 || height(MaxCluster)~=height(MinCluster)
                k2=randperm(height(MinCluster));
                random_val2=MinCluster(k2(1),:);
            else
                random_val2=random_val;
            end
            
            % for j=1:size(Moth_pos,2)                
            %     L(1,j)=L(1,j)+(P(random_val2.Index2,j)-L(1,j))*rand+P(random_val2.Index2,j)-L(1,j)*(2*rand-1);
            % end
            %%%%update L_pos based on selected from random index in minimum cluster
            for j=1:size(Moth_pos,2)                
                L(1,j)=L(1,j)+fw*((P(random_val2.Index2,j)-L(1,j))*(0.2*rand-0.1));
            end
            Flag4ub=L(1,1:dim)>ub;
            Flag4lb=L(1,1:dim)<lb;
            L(1,1:dim)=(L(1,1:dim).*(~(Flag4ub+Flag4lb)))+ub.*Flag4ub+lb.*Flag4lb;            
            [L(1,dim+1),Lgf,Lhf]=feval(fobj,L_pos(1,:),varargin{:});
            [Lviol(1),Lg_res(1,:),Lh_res(1,:)]=violation_velke(varargin{:},Lgf,Lhf,eps_viol,1);
            L(1,dim+2)=Lviol(1);
            L(1,dim+3:dim+5)=Lg_res(1,:);
            L(1,dim+6:dim+11)=Lh_res(1,:);            
    %% ------------------------------------------------------------------------
            %%%%calculate distance of L_pos(update) from all indexes of moth_pos
            update_L_pos_distance=[];
            update_L_pos_similarity=[];
            similarity_dist=[];
            distance_index=[];
            clusters_labels=[];
            for i=1:height(Moth_pos)                                        %Create table based on distance and similarity
                distance_index=[distance_index;i];
                %cluster labels
                variable_k=T2.cluster_label(i);
                clusters_labels=[clusters_labels;variable_k];
                %%%ecludiean distance
                eculidean_dist = pdist2(L(1,1:dim),P(i,1:dim),'minkowski',2);
                update_L_pos_distance=[update_L_pos_distance;eculidean_dist];
                %%%similarity
                similarity_D=pdist2(L(1,1:dim),P(i,1:dim));
                similarity_cosine=1-similarity_D;
                similarity_dist=[similarity_dist;similarity_D];
                update_L_pos_similarity=[update_L_pos_similarity;similarity_cosine];                
            end
            distance_table=table(distance_index,clusters_labels,update_L_pos_distance,similarity_dist,update_L_pos_similarity);
            minimum_distance=min(distance_table.update_L_pos_distance);
            minimum_distance_table=distance_table(distance_table.update_L_pos_distance==minimum_distance,:);
            % here we find the distance and similarity of random index of minimum cluster
            minimum_cluster_min_distance=distance_table(distance_table.distance_index==random_val2.Index2,:);
    %% ------------------------------------------------------------------------
            %%%we find the minimum distances and maximum similarity with index and cluster labels of all cluster(except outlier)
            if height(distance_table)~=0
                %sub_table_cluster_size=[];
                sub_table_minimum_distance=[];
                for i=1:height(unique_idx2)
                    if unique_idx2(i)~=-1
                        %extract the table clusterwise
                        sub_table_distance=distance_table(distance_table.clusters_labels==unique_idx2(i),:);
                        sub_table_minimum=min(sub_table_distance.update_L_pos_distance);
                        variable_j=sub_table_distance(sub_table_distance.update_L_pos_distance==sub_table_minimum,:);
                        sub_table_minimum_distance=[sub_table_minimum_distance;variable_j];
                    end
                end
                ClusterMinD=table(sub_table_minimum_distance);               
            end
    %% ------------------------------------------------------------------------
            %%%%update the cluster (except outlier) accroding to the condition
            if height(ClusterMinD)~=0
                cluster_label_checking=[0];
                for i=1:height(ClusterMinD.sub_table_minimum_distance)
                    sub_part=ClusterMinD.sub_table_minimum_distance(i,:);
                    for j=1:height(sub_part)
                        if sub_part.update_L_pos_similarity(j)>=0.86 & sub_part.update_L_pos_distance(j)<=22.0     %assumption made
                            %store the cluster for checking purpose
                            if any(cluster_label_checking~=sub_part.clusters_labels(j))
                                cluster_label_checking=[cluster_label_checking sub_part.clusters_labels(j)];
                                %update the all values of that cluster in which that index belongs
                                sub_table25= T2(T2.cluster_label==sub_part.clusters_labels(j),:);
                                if height(sub_table25)~=0
                                    %%%%randomly selected index value
                                    k11=randperm(height(sub_table25));
                                    random_value1=sub_table25(k11(1),:);
                                    %%%%upadte the value of moth_pos for all index in a cluster
                                    for i=1:height(sub_table25)
                                        variable_e=sub_table25.Index2(i);
                                        if variable_e~=random_value1.Index2
                                            for j=1:size(Moth_pos,2)
                                                P(variable_e,j)=P(variable_e,j)+(L(1,j)-P(variable_e,j))*(-1+rand)+(P(random_value1.Index2,j)-P(variable_e,j))*(-1+rand);
                                            end
                                            % Check if moths go out of the search space and bring it back
                                            Flag4ub=P(variable_e,1:dim)>ub;
                                            Flag4lb=P(variable_e,1:dim)<lb;
                                            P(variable_e,1:dim)=(P(variable_e,1:dim).*(~(Flag4ub+Flag4lb)))+ub.*Flag4ub+lb.*Flag4lb;
                                            % Calculate the fitness of moths
                                            [P(variable_e,dim+1),gf,hf]=feval(fobj,Moth_pos(variable_e,:),varargin{:});
                                            [newviol(variable_e),newg_res(variable_e,:),newh_res(variable_e,:)]=violation_velke(varargin{:},gf,hf,eps_viol,variable_e);
                                            fval=fval+1;
                                            P(variable_e,dim+2)=newviol(variable_e);
                                            P(variable_e,dim+3:dim+5)=newg_res(variable_e,:);
                                            P(variable_e,dim+6:dim+11)=newh_res(variable_e,:);
                                            fval=fval+1;
                                        end
                                     end
                                end
                            end
                        else
                            if any(cluster_label_checking~=sub_part.clusters_labels(j))
                                cluster_label_checking=[cluster_label_checking sub_part.clusters_labels(j)];
                                %update the all values of that cluster in which that index belongs
                                sub_table25= T2(T2.cluster_label==sub_part.clusters_labels(j),:);
                                for i=1:height(sub_table25.Index2)
                                    variable_w=sub_table25.Index2(i);
                                    for j=1:size(Moth_pos,2)
                                        if variable_w<=Flame_no % Update the position of the moth with respect to its corresponsing flame
                                            % D in Eq. (3.13)
                                            distance_to_flame=abs(sorted_population(variable_w,j)-P(variable_w,j));
                                            b=1;
                                            t=(a-1)*rand+1;
                                            % Eq. (3.12)
                                            P(variable_w,j)=distance_to_flame*exp(b.*t).*cos(t.*2*pi)+sorted_population(variable_w,j);
                                        end
                                        if variable_w>Flame_no % Upaate the position of the moth with respct to one flame
                                            % Eq. (3.13)
                                            distance_to_flame=abs(sorted_population(Flame_no,j)-P(variable_w,j));
                                            b=1;
                                            t=(a-1)*rand+1;
                                            % Eq. (3.12)
                                            P(variable_w,j)=distance_to_flame*exp(b.*t).*cos(t.*2*pi)+sorted_population(Flame_no,j);
                                        end
                                    end
                                    % Check if moths go out of the search space and bring it back
                                    Flag4ub=P(variable_w,1:dim)>ub;
                                    Flag4lb=P(variable_w,1:dim)<lb;
                                    P(variable_w,1:dim)=(P(variable_w,1:dim).*(~(Flag4ub+Flag4lb)))+ub.*Flag4ub+lb.*Flag4lb;
                                    % Calculate the fitness of moths
                                    [P(variable_w,dim+1),gf,hf]=feval(fobj,Moth_pos(variable_w,:),varargin{:});
                                    [newviol(variable_w),newg_res(variable_w,:),newh_res(variable_w,:)]=violation_velke(varargin{:},gf,hf,eps_viol,variable_w);
                                    fval=fval+1;
                                    P(variable_w,dim+2)=newviol(variable_w);
                                    P(variable_w,dim+3:dim+5)=newg_res(variable_w,:);
                                    P(variable_w,dim+6:dim+11)=newh_res(variable_w,:);                                   
                                    fval=fval+1;
                                end
                            end
                        end
                    end
                end
            end
        end 
        %Decide the parameter values of fw
        if Iteration<=(Max_iteration/2)
            lb1=0.6;
            ub1=1.0;
            fw=lb1+(Iteration/Max_iteration);
            if minimum_cluster_min_distance.update_L_pos_distance>=22
                fw=lb1+(Iteration/Max_iteration);
            else
                fw=ub1-((Iteration)/Max_iteration);
            end
        else
            lb1=0.1;
            ub1=0.5;
            fw=lb1+((0.4*Iteration)/Max_iteration);
            if minimum_cluster_min_distance.update_L_pos_distance>=22
                fw=lb1+((0.4*Iteration)/Max_iteration);
            else
                fw=ub1-((0.4*Iteration)/Max_iteration);
            end
        end
        GlobalMins_t(1,Iteration)=Best_flame_score;
        GlobalViol_t(1,Iteration)=POM(1,dim+2);
        Divv(1,Iteration)=(1/N)*(sum(sqrt(sum((Moth_pos-ones(N,1)*mean(Moth_pos)).^2))));
        disp(['Function' num2str(ss) '  Run' num2str(run) ' Iteration ' num2str(Iteration) ': Best Cost = ' num2str(Best_flame_score)]);
        Iteration=Iteration+1;
    end
    GlobalMins_t12(run,:)=GlobalMins_t;
    Globalviol_t12(run,:)=GlobalViol_t;
    [violmin, indexviolmin]=min(P(:,dim+2)); %minimalizuju violation          % based on violation
    pomocna=P(indexviolmin,:);
    [fmin,indexfmin]=min(pomocna(:,dim+1));
    bodmin=pomocna(indexfmin,:);
    viol_fun(run,:)=violmin;
    violFun_min(run,:)=fmin;
    
    [fmin1,indexfmin1]=min(P(:,dim+1));                                       % based on function
    [violmin1, indexviolmin1]=min(P(indexfmin1,dim+2));
    fun_viol(run,:)=violmin1;
    funViol_min(run,:)=fmin1;
    c=spocitejc(bodmin);
    por=length(find(bodmin(dim+3:dim+11)>0));
    fprintf('\n fun=%d run=%d iter=%d ObjVal=%g violmin=%g\n',varargin{:},run,Max_iteration,fmin,violmin);
    Timex=toc;
    vystup=[ss run Iteration violmin fmin violmin1 fmin1 c por fval Timex];
    tabulka=[tabulka;vystup];
    
end
xlswrite(strcat('D:\BHU+Amity\FinalCEC2017\LMFO2017 Results\LMFO 2017 Results\Dim',num2str(dim),'pop',num2str(N),'Run',num2str(runtime),'.xlsx'),[tabulka ], [Function_name 'fitRI']);
xlswrite(strcat('D:\BHU+Amity\FinalCEC2017\LMFO2017 Results\LMFO 2017 Results\Dim',num2str(dim),'pop',num2str(N),'Final.xlsx'),[num mean(tabulka(:,5)) std(tabulka(:,5)) min(tabulka(:,5)) max(tabulka(:,5)) mean(tabulka(:,4)) min(tabulka(:,4)) mean(tabulka(:,7)) std(tabulka(:,7)) min(tabulka(:,7)) max(tabulka(:,7)) mean(tabulka(:,6)) min(tabulka(:,6)) mean(tabulka(:,12)) mean(tabulka(:,13))],Function_name);

xlswrite(strcat('D:\BHU+Amity\FinalCEC2017\LMFO2017 Results\LMFO 2017 Results\Dim',num2str(dim),'LMFOconvergenceR.xlsx'),[GlobalMins_t12 ], [Function_name 'fitRI']);

xlswrite(strcat('D:\BHU+Amity\FinalCEC2017\LMFO2017 Results\LMFO 2017 Results\LMFO_@D',num2str(dim),'pop',num2str(N),'violFun.xlsx'),[viol_fun],Function_name);
xlswrite(strcat('D:\BHU+Amity\FinalCEC2017\LMFO2017 Results\LMFO 2017 Results\LMFO_@D',num2str(dim),'pop',num2str(N),'violFunMin.xlsx'),[violFun_min],Function_name);

xlswrite(strcat('D:\BHU+Amity\FinalCEC2017\LMFO2017 Results\LMFO 2017 Results\LMFO_@D',num2str(dim),'pop',num2str(N),'funViol.xlsx'),[fun_viol],Function_name);
xlswrite(strcat('D:\BHU+Amity\FinalCEC2017\LMFO2017 Results\LMFO 2017 Results\LMFO_@D',num2str(dim),'pop',num2str(N),'funViolMin.xlsx'),[funViol_min],Function_name);

xlswrite(strcat('D:\BHU+Amity\FinalCEC2017\LMFO2017 Results\LMFO 2017 Results\LMFO_@D',num2str(dim),'pop',num2str(N),'funMin.xlsx'),[GlobalMins_t12],Function_name);
xlswrite(strcat('D:\BHU+Amity\FinalCEC2017\LMFO2017 Results\LMFO 2017 Results\MFO_@D',num2str(dim),'pop',num2str(N),'ViolMin.xlsx'),[Globalviol_t12],Function_name);

viol_fun=[];
fun_viol=[];
violFun_min=[];
funViol_min=[];
save all
end