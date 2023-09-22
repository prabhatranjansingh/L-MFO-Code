% function [Best_flame_score,Best_flame_pos,Convergence_curve]=MFO(N,Max_iteration,lb,ub,dim,fobj,num,fmin)
% function [Best_flame_score,Feval,GlobalMins_t12]= MFO(fobj,Max_iteration,Max_FES,N,dim,lb,ub,fmin,varargin)
function [Best_flame_pos,Best_flame_score,Feval]= L_MFO_1(fobj,Max_iteration,Max_FES,N,dim,lb,ub,varargin)
num=cell2mat(varargin);
ss=num2str(num);
Function_name=strcat('F',ss);
display('MFO is optimizing your problem');
runtime=30;
UB=ones(1,dim).*ub;
LB=ones(1,dim).*lb;
% GlobalMins_t12=zeros(runtime,Max_iteration);
for run=1:runtime
     tic 
     succ_rate=0;
     fval=0;
     Feval=0;
    %Initialize the positions of moths
    Moth_pos=initialization(N,dim,ub,lb);
    L_pos=initialization(1,dim,ub,lb);
    Convergence_curve=zeros(1,Max_iteration);
    GlobalMins_t=zeros(1,Max_iteration);
    Iteration=1;
    record=zeros(Max_iteration,N);
    %%%%%%%%%%%%%%%%%%%% Let we take the value of minpts and maxpts as user input
    minpts=3;
    maxpts=size(Moth_pos,1);
    Toutliers=[];
    fw=1;
    % Main loop
    while Iteration<Max_iteration+1
        % Number of flames Eq. (3.14) in the paper
        Flame_no=round(N-Iteration*((N-1)/Max_iteration));
        Flag4ub=L_pos(1,:)>ub;
        Flag4lb=L_pos(1,:)<lb;
        L_pos(1,:)=(L_pos(1,:).*(~(Flag4ub+Flag4lb)))+ub.*Flag4ub+lb.*Flag4lb;
        L_fitness=fobj(L_pos(1,:));
        for i=1:size(Moth_pos,1)        
            % Check if moths go out of the search spaceand bring it back
            Flag4ub=Moth_pos(i,:)>ub;
            Flag4lb=Moth_pos(i,:)<lb;
            Moth_pos(i,:)=(Moth_pos(i,:).*(~(Flag4ub+Flag4lb)))+ub.*Flag4ub+lb.*Flag4lb;  
            % Calculate the fitness of moths
    %         Moth_fitness(1,i)=fobj(Moth_pos(i,:));  
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
        % Update the position best flame obtained so far
        Best_flame_score=fitness_sorted(1);
        Best_flame_pos=sorted_population(1,:);

        previous_population=Moth_pos;
        previous_fitness=Moth_fitness;
        % a linearly dicreases from (-1 to -2) to calculate t in Eq. (3.12)
        a=-1+Iteration*((-1)/Max_iteration);  
        %% disp("MOTH POSITION CODING");
        %% Moth position
        %%here we store the index into a list
        Index2=[1 : size(Moth_pos,1)]';                                     % 1,2,3,..... 30(Moth size)
        %%Now we apply the DBSCAN Clustering algroithm
        %%%%%%%%%%%%%%%%%%%%find the epsilon value
        epsilon2=3;
%         epsilon2=clusterDBSCAN.estimateEpsilon(Moth_pos,minpts,maxpts);        
% %         clusterDBSCAN.estimateEpsilon(Moth_pos,minpts,maxpts);
        %%%%%%%%%%%%%%%%%%%After getting the value of epsilon we impelemt the
        %%%%%%%%%%%%%%%%%%%dbscan alogrithm%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        cluster_label=dbscan(Moth_pos,epsilon2,minpts);
        unique_idx2=unique(cluster_label);
        %here we add the labels into data frame
        T2=table(Index2,Moth_pos,cluster_label);
        %%%Now we find each cluster.%%%
        Cluster_Labels2=[];
        Size_of_each_cluster2=[];
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
                maximum_table2=tail(maximum_table2,1);
                %(ii) Here we store the index of value which belong into that particular cluster
                for i=1:length(unique_idx2)
                    if unique_idx2(i)==maximum_table2.Cluster_Labels2
                        MaxCluster=T2(T2.cluster_label==maximum_table2.Cluster_Labels2,:);
                    end
                end
            end
    %% ------------------------------------------------------------------------
            %Randomly select individual from (Heighest size) cluster
            k=randperm(height(MaxCluster));
            random_val=MaxCluster(k(1),:);
            %%%update the index of the outliers towards the maximum cluster
            if height(Toutliers)~=0
                for i=1:height(Toutliers)
                    %% update outliers towards the maximum cluster
                    K_P(1,:)=rand.*Moth_pos(Toutliers.Index2(i),:)+(rand.*Moth_pos(Toutliers.Index2(i),:)-Moth_pos(Toutliers.Index2(i),:)).*(-1+2*rand)+...
                            rand.*(Moth_pos(random_val.Index2,:)-Moth_pos(Toutliers.Index2(i),:));
                    K_Q(1,:)=(UB+LB-Moth_pos(Toutliers.Index2(i),:));
                    K_P_fitness(1)=fobj(K_P(1,:));
                    K_Q_fitness(1)=fobj(K_Q(1,:));
                    if K_P_fitness(1)<K_Q_fitness(1)
                       Moth_pos(Toutliers.Index2(i),:)= K_P(1,:);
                       Moth_fitness(1,Toutliers.Index2(i))=K_P_fitness(1);
                    else
                       Moth_pos(Toutliers.Index2(i),:)= K_Q(1,:);
                       Moth_fitness(1,Toutliers.Index2(i))=K_Q_fitness(1);
                    end
                    fval=fval+2;
                end
            end
            %Randomly select individual from (Lowest size) cluster
            if height(Noutliers)~=1 || height(MaxCluster)~=height(MinCluster)
                k2=randperm(height(MinCluster));
                random_val2=MinCluster(k2(1),:);
            else
                random_val2=random_val;
            end
    %% ------------------------------------------------------------------------
% %             %%%%find the distance between L_pos(before) with all indexes of minimum cluster
% %             L_pos_distance=[];
% %             L_pos_similarity=[];
% %             distance_index1=[];
% %             for i=1:height(MinCluster)
% %                 distance_index1=[distance_index1;MinCluster.Index2(i)];
% %                 %%%ecludiean distance
% %                 eculidean_dist1 = pdist2(L_pos,Moth_pos(MinCluster.Index2(i),:),'minkowski',2);
% %                 L_pos_distance=[L_pos_distance;eculidean_dist1];
% %                 %%%%similarity
% %                 similarity_dist1=pdist2(L_pos,Moth_pos(MinCluster.Index2(i),:),'cosine');
% %                 similarity_1=1-similarity_dist1;
% %                 L_pos_similarity=[L_pos_similarity;similarity_1];
% %             end
% %             distance_table1=table(distance_index1,L_pos_distance,L_pos_similarity);
% %             minimum_distance1=min(distance_table1.L_pos_distance);
% %             minimum_distance_table1=distance_table1(distance_table1.L_pos_distance==minimum_distance1,:);
%                 for i=1:size(Moth_pos,1)
%                     for j=1:size(Moth_pos,1)
%                         if i~=j
%                             Ed=pdist2(Moth_pos(i,:),Moth_pos(j,:),'minkowski',2);
%                         end
%                     end
%                 end
    %% ------------------------------------------------------------------------
            %%%%UPDATE L_POS%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %%%%update L_pos based on selected from random index in minimum cluster
            for j=1:size(Moth_pos,2)
                %%%%%L_pos=L_pos+rand*(Moth_pos(random_val.Index2,:)-L_pos);
                L_pos(1,j)=L_pos(1,j)+fw*((Moth_pos(random_val2.Index2,j)-L_pos(1,j))*(0.2*rand-0.1));
%                 L_pos(1,j)=L_pos(1,j)+((Moth_pos(random_val2.Index2,j)-L_pos(1,j))*(0.6*rand-0.3));
            end
            L_fitness=fobj(L_pos(1,:));
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
                eculidean_dist = pdist2(L_pos,Moth_pos(i,:),'minkowski',2);
                update_L_pos_distance=[update_L_pos_distance;eculidean_dist];
                %%%similarity
                similarity_D=pdist2(L_pos,Moth_pos(i,:),'cosine');
                similarity_cosine=1-similarity_D;
                similarity_dist=[similarity_dist;similarity_D];
                update_L_pos_similarity=[update_L_pos_similarity;similarity_cosine];                
            end
            distance_table=table(distance_index,clusters_labels,update_L_pos_distance,similarity_dist,update_L_pos_similarity);
            minimum_distance=min(distance_table.update_L_pos_distance);
            minimum_distance_table=distance_table(distance_table.update_L_pos_distance==minimum_distance,:);
            % here we find the distance and similarity of random index of minimum cluster
            minimum_cluster_min_distance=distance_table(distance_table.distance_index==random_val2.Index2,:);
            %%%compare the similarity and distance of L_pos vs L_pos(update)for a random index of minimum cluster before update L_pos
    % % % % % % % % % % % % %         sub_table26=distance_table1(distance_table1.distance_index1==random_val2.Index2,:);
    % % % % % % % % % % % % %         sub_table27=distance_table(distance_table.distance_index==random_val2.Index2,:);
    % % % % % % % % % % % % %         compare_table=table(sub_table26,sub_table27);
    %% ------------------------------------------------------------------------
            %%%we find the minimum distances and maximum similarity with index and cluster labels of all cluster(except outlier)
            if height(distance_table)~=0
                %sub_table_cluster_size=[];
                sub_table_minimum_distance=[];
                for i=1:height(unique_idx2)
                    if unique_idx2(i)~=-1
                        %extract the table clusterwise
                        sub_table_distance=distance_table(distance_table.clusters_labels==unique_idx2(i),:);
                        %sub_table_cluster_size=[sub_table_cluster_size;height(sub_table_distance)];
                        %extract the distances and similarity of the these index of particular cluster and find the index and cluster_label which having minimum distance and maximum similarity of each cluster
                        sub_table_minimum=min(sub_table_distance.update_L_pos_distance);
                        variable_j=sub_table_distance(sub_table_distance.update_L_pos_distance==sub_table_minimum,:);
                        sub_table_minimum_distance=[sub_table_minimum_distance;variable_j];
                    else
                       %fprintf("Only outlier present")
                    end
                end
                ClusterMinD=table(sub_table_minimum_distance);
                %ClusterMinD=table(sub_table_cluster_size,sub_table_minimum_distance);
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
                                                Moth_pos(variable_e,j)=Moth_pos(variable_e,j)+(L_pos(1,j)-Moth_pos(variable_e,j))*(-1+rand)+(Moth_pos(random_value1.Index2,j)-Moth_pos(variable_e,j))*(-1+rand);
                                            end
                                            % Check if moths go out of the search space and bring it back
                                            Flag4ub=Moth_pos(variable_e,:)>ub;
                                            Flag4lb=Moth_pos(variable_e,:)<lb;
                                            Moth_pos(variable_e,:)=(Moth_pos(variable_e,:).*(~(Flag4ub+Flag4lb)))+ub.*Flag4ub+lb.*Flag4lb;
                                            % Calculate the fitness of moths
                                            %Moth_fitness(1,i)=fobj(Moth_pos(i,:));
                                            Moth_fitness(1,variable_e)=fobj(Moth_pos(variable_e,:));
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
                                            distance_to_flame=abs(sorted_population(variable_w,j)-Moth_pos(variable_w,j));
                                            b=1;
                                            t=(a-1)*rand+1;
                                            % Eq. (3.12)
                                            Moth_pos(variable_w,j)=distance_to_flame*exp(b.*t).*cos(t.*2*pi)+sorted_population(variable_w,j);
                                        end
                                        if variable_w>Flame_no % Upaate the position of the moth with respct to one flame
                                            % Eq. (3.13)
                                            distance_to_flame=abs(sorted_population(Flame_no,j)-Moth_pos(variable_w,j));
                                            b=1;
                                            t=(a-1)*rand+1;
                                            % Eq. (3.12)
                                            Moth_pos(variable_w,j)=distance_to_flame*exp(b.*t).*cos(t.*2*pi)+sorted_population(Flame_no,j);
                                        end
                                    end
                                    Moth_fitness(1,variable_w)=fobj(Moth_pos(variable_w,:));
                                    fval=fval+1;
                                end
                            end
                        end
                    end
                end
            end
        end 
    %% ------------------------------------------------------------------------
        record(Iteration,:)=fitness_sorted;
        moth_record(:,:)=sorted_population;  
        Convergence_curve(Iteration)=Best_flame_score;    
        % Display the iteration and best optimum obtained so far
    %     if mod(Iteration,50)==0
        %display(['At iteration ', num2str(Iteration), ' the best fitness is ', num2str(Best_flame_score)]);
        display(['Function ',ss,'At Run ',num2str(run),'iteration ', num2str(Iteration), ' the best fitness is ', num2str(Best_flame_score)]);
    %     end
        GlobalMins_t(1,Iteration)=Best_flame_score;
        Divv(1,Iteration)=(1/N)*(sum(sqrt(sum((Moth_pos-ones(N,1)*mean(Moth_pos)).^2))));
        if Iteration<=(Max_iteration/2)
            lb1=0.6;
            ub1=1.0;
            if minimum_cluster_min_distance.update_L_pos_distance>=22
                fw=lb1+(Iteration/Max_iteration);
            else
                fw=ub1-((Iteration)/Max_iteration);
            end
        else
            lb1=0.1;
            ub1=0.5;
            if minimum_cluster_min_distance.update_L_pos_distance>=22
                fw=lb1+((0.4*Iteration)/Max_iteration);
            else
                fw=ub1-((0.4*Iteration)/Max_iteration);
            end
        end
        Iteration=Iteration+1;
    end
    records(run,:,:)=record;
    Diversity(run,:)= Divv;
    runn(run)=run;
    Global_min(num,run)=Best_flame_score;
    GlobalMins(run)=Best_flame_score;
    minimum=min(GlobalMins_t);
    StandarDev(run)=std(GlobalMins_t);    
    %Mean_run(run)=mean(GlobalMins_t);          
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
xlswrite('L_MFO1DiversityD_500_N_30_runtime_30.xlsx',[Diversity ], [Function_name 'Divr']);
xlswrite('L_MFO1ConvergeD_500_pop_30Run_30.xlsx',[GlobalMins_t12 ], [Function_name 'fitRI']);
xlswrite('L_MFO1RunD_500_Pop_30_Run.xlsx',[runn' GlobalMins' TFE' Timex'],[Function_name],'A2');
xlswrite('L_MFO1FinalD_500_pop_30Run.xlsx',[num mean(Global_min(num,:)) std(Global_min(num,:)) min(Global_min(num,:)) max(Global_min(num,:)) TFEval(num) Time(num)],Function_name);
save all
end