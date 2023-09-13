% function [Best_flame_score,Best_flame_pos,Convergence_curve]=MFO(N,Max_iteration,lb,ub,dim,fobj,num,fmin)
% function [Best_flame_score,Feval,GlobalMins_t12]= MFO(fobj,Max_iteration,Max_FES,N,dim,lb,ub,fmin,varargin)
function [Best_flame_pos,Best_flame_score,Feval]= L_MFO(fobj,Max_iteration,Max_FES,N,dim,lb,ub,varargin)
num=cell2mat(varargin);
ss=num2str(num);
Function_name=strcat('F',ss);
display('MFO is optimizing your problem');
runtime=1;
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
prompt="Enter the minpts value";
minpts=input(prompt);
disp(minpts);

prompt="Enter the maxpts value";
maxpts=input(prompt);
disp(maxpts);

% Main loop
while Iteration<Max_iteration+1
    
    % Number of flames Eq. (3.14) in the paper
    Flame_no=round(N-Iteration*((N-1)/Max_iteration));
    Flag4ub=L_pos(1,:)>ub;
    Flag4lb=L_pos(1,:)<lb;
    L_pos(1,:)=(L_pos(1,:).*(~(Flag4ub+Flag4lb)))+ub.*Flag4ub+lb.*Flag4lb;
    L_fitness=feval(fobj,L_pos(1,:),varargin{:});
    for i=1:size(Moth_pos,1)        
        % Check if moths go out of the search spaceand bring it back
        Flag4ub=Moth_pos(i,:)>ub;
        Flag4lb=Moth_pos(i,:)<lb;
        Moth_pos(i,:)=(Moth_pos(i,:).*(~(Flag4ub+Flag4lb)))+ub.*Flag4ub+lb.*Flag4lb;  
        
        % Calculate the fitness of moths
%         Moth_fitness(1,i)=fobj(Moth_pos(i,:));  
        Moth_fitness(1,i)=feval(fobj,Moth_pos(i,:),varargin{:});
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
    %......................................................................
%     disp("MOTH POSITION CODING");
    %% Moth position
    %%here we store the index into a list
    Index2=[1 : size(Moth_pos,1)]';
    %%Now we apply the DBSCAN Clustering algroithm
    %%%%%%%%%%%%%%%%%%%%find the epsilon value
    epsilon2=clusterDBSCAN.estimateEpsilon(Moth_pos,minpts,maxpts);
% %     disp("Epsilon value be")
% %      epsilon2;
    %%%%%%%%%%%%%%%%%%%After getting the value of epsilon we impelemt the
    %%%%%%%%%%%%%%%%%%%dbscan alogrithm%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    idx2=dbscan(Moth_pos,epsilon2,minpts);
%     disp("Size of idx be");
    size(idx2);
    unique_idx2=unique(idx2);
%     disp("Length of unique idx value be");
%     disp(length(unique_idx2));
    
    %here we add the labels into data frame
    T2=table(Index2,Moth_pos,idx2);
    %%%Now we find each cluster.%%%
    n21=[];
    n22=[];
    for i=1:height(unique_idx2)
        n21=[n21;unique_idx2(i)];
        %print the subtable for each cluster_labels%
        sub_table21=T2(T2.idx2==unique_idx2(i),:);
        %size of each cluster
        n22=[n22;height(sub_table21)];
    end
    Cluster_Labels2=n21;
    Size_of_each_cluster2=n22;
    dataframe2=table(Cluster_Labels2,Size_of_each_cluster2);
    dataframe2;
    
    %%%%%%%%%%%For Outlier Work%%%%%%%%%%%%%%
    if any(dataframe2.Cluster_Labels2==-1)
        %(i) Find the indexes of Outliers and store into the excel
        for i=1:length(unique_idx2)
            if unique_idx2(i)==-1
                sub_table22=T2(T2.idx2==-1,:);
            else
                continue
            end
        end
        sub_table22;
%         disp("length of index be");
%         disp(height(sub_table22));
    else
%         fprintf("No Outlier Present")
    end
    %%%%%%%%%%%%%%%%%%%%%For Non Outlier Work%%%%%%%%%%%%%%%%%%%%%%
    sub_table23=dataframe2(dataframe2.Cluster_Labels2~=-1,:);
    sub_table23;
    
    %%%%%%%%%%%%maximum and minimum cluster find%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if height(sub_table23)~=0
        %%%%%%%%%%%%%%%%%%%%%%%minimum cluster%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %lowest size
        minimum_size2=min(sub_table23.Size_of_each_cluster2);
        minimum_table2=sub_table23(sub_table23.Size_of_each_cluster2==minimum_size2,:);
        %%here we check that wheather more than one cluster comes under minimum cluster tag or not
        if height(minimum_table2)>=1
            minimum_table2=head(minimum_table2,1);
            minimum_table2;
            %(ii) Here we store the index of value which belong into that particular cluster
            for i=1:length(unique_idx2)
                if unique_idx2(i)==minimum_table2.Cluster_Labels2
                    sub_table29=T2(T2.idx2==minimum_table2.Cluster_Labels2,:);
                else
                    continue
                end
            end
            sub_table29;
        end
        %%%%%%%%%%%%%Maximum cluster%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Heighest size
        maximum_size2=max(sub_table23.Size_of_each_cluster2);
        maximum_table2=sub_table23(sub_table23.Size_of_each_cluster2==maximum_size2,:);
        %%here we check that wheather more than one cluster comes under maximum cluster tag or not
        if height(maximum_table2)>=1
            maximum_table2=head(maximum_table2,1);
            maximum_table2;
            %(ii) Here we store the index of value which belong into that particular cluster
            for i=1:length(unique_idx2)
                if unique_idx2(i)==maximum_table2.Cluster_Labels2
                    sub_table24=T2(T2.idx2==maximum_table2.Cluster_Labels2,:);
                else
                    continue
                end
            end
            sub_table24;
        else
%             fprintf("No such cluster formed")
        end 
        %Randomly select individual from (Heighest size) cluster
        if height(sub_table24)~=0
            k=randperm(height(sub_table24));
            random_val=sub_table24(k(1),:);
            %%%update the index of the outliers towards the maximum cluster
            if height(sub_table22)~=0
                for i=1:height(sub_table22)
                    %% update towards the maximum cluster
                    Moth_pos(sub_table22.Index2(i),:)=rand.*Moth_pos(sub_table22.Index2(i),:)+(rand.*Moth_pos(sub_table22.Index2(i),:)-Moth_pos(sub_table22.Index2(i),:)).*(-1+2*rand)+...
                        rand.*(Moth_pos(random_val.Index2,:)-Moth_pos(sub_table22.Index2(i),:));
                    Moth_fitness(1,sub_table22.Index2(i))=feval(fobj,Moth_pos(sub_table22.Index2(i),:),varargin{:});
                    fval=fval+1;
                end                
            else
%                 fprintf("No outlier present in the data")
            end
            %%%%find the distance between L_pos(before) with all indexes of maximum cluster
            L_pos_distance=[];
            L_pos_similarity=[];
            distance_index1=[];
            for i=1:height(sub_table24)
                distance_index1=[distance_index1;sub_table24.Index2(i)];
                %%%ecludiean distance
                eculidean_dist1 = pdist2(L_pos,Moth_pos(sub_table24.Index2(i),:),'minkowski',2);
                L_pos_distance=[L_pos_distance;eculidean_dist1];
                %%%%similarity
                similarity_dist1=pdist2(L_pos,Moth_pos(sub_table24.Index2(i),:),'cosine');
                similarity_1=1-similarity_dist1;
                L_pos_similarity=[L_pos_similarity;similarity_1];
            end
            %distance_index1=transpose(distance_index1);
            %L_pos_distance=transpose(L_pos_distance);
            %L_pos_similarity=transpose(L_pos_similarity);
            distance_table1=table(distance_index1,L_pos_distance,L_pos_similarity);
            minimum_distance1=min(distance_table1.L_pos_distance);
            minimum_distance_table1=distance_table1(distance_table1.L_pos_distance==minimum_distance1,:);
            
            %%%%UPDATE L_POS%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %%%%update L_pos based on selected from random index in maximum cluster
            for j=1:size(Moth_pos,2)
                %%%%%L_pos=L_pos+rand*(Moth_pos(random_val.Index2,:)-L_pos);
                L_pos(1,j)=L_pos(1,j)+ (Moth_pos(random_val.Index2,j)-L_pos(1,j))*rand;
            end
            L_fitness=feval(fobj,L_pos(1,:),varargin{:});
            %%%%calculate distance of L_pos(update) from all indexes of moth_pos
            update_L_pos_distance=[];
            update_L_pos_similarity=[];
            distance_index=[];
            clusters_labels=[];
            for i=1:height(Moth_pos)
                distance_index=[distance_index;i];
                %cluster labels
                variable_k=T2.idx2(i);
                clusters_labels=[clusters_labels;variable_k];
                %%%ecludiean distance
                eculidean_dist = pdist2(L_pos,Moth_pos(i,:),'minkowski',2);
                update_L_pos_distance=[update_L_pos_distance;eculidean_dist];
                %%%similarity
                similarity_dist=pdist2(L_pos,Moth_pos(i,:),'cosine');
                similarity_cosine=1-similarity_dist;
                update_L_pos_similarity=[update_L_pos_similarity;similarity_cosine];
            end
            %distance_index=transpose(distance_index);
            %update_L_pos_distance=transpose(update_L_pos_distance);
            %update_L_pos_similarity=transpose(update_L_pos_similarity);
            distance_table=table(distance_index,clusters_labels,update_L_pos_distance,update_L_pos_similarity);
            minimum_distance=min(distance_table.update_L_pos_distance);
            minimum_distance_table=distance_table(distance_table.update_L_pos_distance==minimum_distance,:);
            %now we find the cluster in which this index is belongs
            variable_c=minimum_distance_table.distance_index;
            for i=1:height(T2)
                minimum_distance_cluster_table=T2(T2.Index2==variable_c,:);
            end
            minimum_distance_cluster_table;
            %%%compare the similarity and distance of L_pos vs L_pos(update)for a random index of maximum cluster before update L_pos
            sub_table26=distance_table1(distance_table1.distance_index1==random_val.Index2,:);
            sub_table27=distance_table(distance_table.distance_index==random_val.Index2,:);
            compare_table=table(sub_table26,sub_table27);
            
            %%%we find the minimum distances and maximum similarity with index and cluster labels of all cluster(except outlier)
            sub_table_cluster_size=[];
            sub_table_minimum_distance=[];
            for i=1:height(unique_idx2)
                if unique_idx2(i)~=-1
                    %extract the table clusterwise
                    sub_table_distance=distance_table(distance_table.clusters_labels==unique_idx2(i),:);
                    sub_table_cluster_size=[sub_table_cluster_size;height(sub_table_distance)];
                    %extract the distances and similarity of the these index of particular cluster and find the index and cluster_label which having minimum distance and maximum similarity of each cluster
                    sub_table_minimum=min(sub_table_distance.update_L_pos_distance);
                    variable_j=sub_table_distance(sub_table_distance.update_L_pos_distance==sub_table_minimum,:);
                    sub_table_minimum_distance=[sub_table_minimum_distance;variable_j];
                else
%                     fprintf("Only outlier present")
                end
            end
            dataframe3=table(sub_table_cluster_size,sub_table_minimum_distance);
            %%%%update the cluster (except outlier) accroding to the condition
            if height(dataframe3)~=0
                cluster_label_checking=[0];
                for i=1:height(dataframe3.sub_table_minimum_distance)
                    sub_part=dataframe3.sub_table_minimum_distance(i,:);
                    for j=1:height(sub_part)
                        if sub_part.update_L_pos_similarity(j)>=0.86 & sub_part.update_L_pos_distance(j)<=22.0
                            %store the cluster for checking purpose
                            if any(cluster_label_checking~=sub_part.clusters_labels(j))
                                cluster_label_checking=[cluster_label_checking sub_part.clusters_labels(j)];
                                %update the all values of that cluster in which that index belongs
                                sub_table25= T2(T2.idx2==sub_part.clusters_labels(j),:);
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
                                            % Check if moths go out of the search spaceand bring it back
                                            Flag4ub=Moth_pos(variable_e,:)>ub;
                                            Flag4lb=Moth_pos(variable_e,:)<lb;
                                            Moth_pos(variable_e,:)=(Moth_pos(variable_e,:).*(~(Flag4ub+Flag4lb)))+ub.*Flag4ub+lb.*Flag4lb;
                                            % Calculate the fitness of moths
                                            %Moth_fitness(1,i)=fobj(Moth_pos(i,:));
                                            Moth_fitness(1,variable_e)=feval(fobj,Moth_pos(variable_e,:),varargin{:});
                                            fval=fval+1;
                                        end
                                    end
                                end
                            else
                                continue
                            end
                        else
                            %%fprintf("Condition not statisfies")
                            if any(cluster_label_checking~=sub_part.clusters_labels(j))
                                cluster_label_checking=[cluster_label_checking sub_part.clusters_labels(j)];
                                %update the all values of that cluster in which that index belongs
                                sub_table25= T2(T2.idx2==sub_part.clusters_labels(j),:);
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
                                            distance_to_flame=abs(sorted_population(variable_w,j)-Moth_pos(variable_w,j));
                                            b=1;
                                            t=(a-1)*rand+1;
                                            
                                            % Eq. (3.12)
                                            Moth_pos(variable_w,j)=distance_to_flame*exp(b.*t).*cos(t.*2*pi)+sorted_population(Flame_no,j);
                                        end
                                    end
                                    Moth_fitness(1,variable_w)=feval(fobj,Moth_pos(variable_w,:),varargin{:});
                                    fval=fval+1;
                                end
                            else
                                continue
                            end
                        end
                    end
                end
            else
%                 fprintf("No updation possible")
            end
        else
%             fprintf("No maximum cluster formed, so we cannt find it")
        end
    else
%         fprintf("Only Outlier Present in dataframe")
    end 
    record(Iteration,:)=fitness_sorted;
    moth_record(:,:)=sorted_population;  
    Convergence_curve(Iteration)=Best_flame_score;    
    % Display the iteration and best optimum obtained so far
%     if mod(Iteration,50)==0
    display(['At iteration ', num2str(Iteration), ' the best fitness is ', num2str(Best_flame_score)]);
%     end
    GlobalMins_t(1,Iteration)=Best_flame_score;
    Divv(1,Iteration)=(1/N)*(sum(sqrt(sum((Moth_pos-ones(N,1)*mean(Moth_pos)).^2))));
    if Iteration<30
        A=Iteration;
        sheet=A;
        %xlswrite('MothRecord1.xlsx',[moth_record],sheet);
    end
    Iteration=Iteration+1;
end
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
%%time(run)=toc;
%%GlobalMins(run)=Best_flame_score;
%%Global_min(num,run)=Best_flame_score;
%%GlobalMins_t12(run,:)=GlobalMins_t;
%%Run(run)=run;
%%TFE(run)=Feval;
%%Feval1(run)=fval;
% % fmin1(run)=fmin;
% % Success_run(run)=succ_rate;
% %     if Feval>=1
% %         Feval2(run)=Feval;
% %     else
% %         Feval2(run)=fval;
% %     end
end
% Success(num)=sum(Success_run);
        
        Time(num)=mean(Timex);
        TFEval(num)=mean(TFE);      
%         xlswrite('MFO10D30Pop30RunDiversity.xlsx',[Diversity ], [Function_name 'Divr']);

%         xlswrite('MFORun10D30pop30Run.xlsx',[GlobalMins_t12 ], [Function_name 'fitRI']);
%         xlswrite('MFO10D30PopRun.xlsx',[runn' Mean_run' StandarDev' GlobalMins' TFE' Timex'],[Function_name],'A2');
%         xlswrite('MFOFinal10D30pop30Run.xlsx',[num mean(Global_min(num,:)) std(Global_min(num,:)) min(Global_min(num,:)) max(Global_min(num,:)) TFEval(num) Time(num)],Function_name);
save all
end
