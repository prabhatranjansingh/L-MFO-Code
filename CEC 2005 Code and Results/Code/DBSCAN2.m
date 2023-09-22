%%%%%%Now we take the user input for particular iteration%%
prompt="Enter the iteration";
iteration=input(prompt);
disp(iteration)


%%Here we the read the data for particular iteration%%
prompt="Enter the Sheetname";
sheet=input(prompt,"s");
disp(sheet)
data=xlsread('D:\MFO1\MothRecord1.xlsx',sheet);
data(:,:)=data;
disp('Size of Data be:')
size(data)


%%data=transpose(data);
%%disp("size of the original dataframe be")
%%size(data)



%%we store the data from df for the particular iteration into andother df
%%data1=data(:,iteration);
%%disp("Size of the dataframe for particular iteration")
%%size(data1)

%%here we store the index into a list
Index=[];
for i=1:height(data)
    Index=[Index i];
end
Index=transpose(Index);
disp("size of the index of data")
size(Index)

%%we draw the scatter plot of new dataframe%%
plot(Index,data,'r*')
xlabel('index')
ylabel('value')
title('Scatter Plot')



%%Now we apply the DBSCAN Clustering algroithm
%%%%%%%%%%%%%%%%%%%% Let we take the value of minpts and maxpts as user input
prompt="Enter the minpts value"
minpts=input(prompt);
disp(minpts)

prompt="Enter the maxpts value"
maxpts=input(prompt);
disp(maxpts)


%%%%%%%%%%%%%%%%%%%%find the epsilon value
epsilon=clusterDBSCAN.estimateEpsilon(data,minpts,maxpts);
disp("Epsilon value be")
epsilon


%%%%%%%%%%%%%%%%%%%After getting the value of epsilon we impelemt the
%%%%%%%%%%%%%%%%%%%dbscan alogrithm%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[idx,corepts]=dbscan(data,epsilon,minpts);
disp("Size of idx be")
size(idx)
unique_idx=unique(idx)
disp("Length of unique idx value be")
disp(length(unique_idx))
disp(corepts)

%here we add the labels into data frame
T=table(Index,data,idx);
T(1:5,:)

%%%Now we find the centroid of each cluster.%%%
n1=[];
n2=[];
centroid=[];
overall_centroid=[];
for i=1:height(unique_idx)
    n1=[n1 unique_idx(i)];
    
    %print the subtable for each cluster_labels%
    sub_table=T(T.idx==unique_idx(i),:)
    
end
sub_table
    n2=[n2 height(sub_table)];
    
    %find the mean of the each cluster
    mean_value=mean(sub_table.data);
    %centroid=[centroid mean_value];
    overall_centroids=mean(mean_value);
    overall_centroid=[overall_centroid overall_centroids]
%end
 %display the value of n1, n2 and centroid and assign in a single table%

 Cluster_Labels=transpose(n1);
 Size_of_each_cluster=transpose(n2);
 overall_centroid=transpose(overall_centroid);
 Cluster_Labels;
 Size_of_each_cluster;
 overall_centroid;
dataframe=table(Cluster_Labels,Size_of_each_cluster,overall_centroid);
dataframe


%%%%%%%%%%%For Outlier Work%%%%%%%%%%%%%%
%(i) Find the centroid of outlier and store into excel%
 outlier_centroid=dataframe.overall_centroid(dataframe.Cluster_Labels==-1);
 disp("Outlier centroid be")
 outlier_centroid
 %(ii) Find the indexes of Outliers and store into the excel
 for i=1:length(unique_idx)
     if unique_idx(i)==-1
         sub_table1=T(T.idx==-1,:);    
     end
end
sub_table1
disp("length of index be")
disp(height(sub_table1))
    



%%%%%%%%%%%%%%%%%%%%%For Non Outlier Work%%%%%%%%%%%%%%%%%%%%%%

%(i) Find the maximum value of the centroid of the clusters(except outlier cluster)
sub_table2=dataframe(dataframe.Cluster_Labels~=-1,:);
sub_table2

maximum_centroid_value=max(sub_table2.overall_centroid);
disp("Maximum centroid be")
maximum_centroid_value
maximum_table=sub_table2(sub_table2.overall_centroid==maximum_centroid_value,:);
maximum_table

%(ii) Here we store the index of value which belong into that particular cluster
for i=1:length(unique_idx)
    if unique_idx(i)==maximum_table.Cluster_Labels
        sub_table3=T(T.idx==maximum_table.Cluster_Labels,:);    
    end
end
sub_table3
disp("length of index be")
disp(height(sub_table3))


%(iii) Find closest value and index to the centroid of that cluster%
for i=1:height(sub_table3)
    for j=1:length(sub_table.data)
        distance=pdist2(maximum_centroid_value,sub_table3.data(i,j));
        distance_matrix(i,j)=distance;
    end
end
%eculidean_distances
%eculidean_distance=transpose(eculidean_distances);
%eculidean_distance;
disp("size of distance_matrix")
size(distance_matrix)

sub_table4=table(sub_table3,distance_matrix);

%Now we find the our objective in this question%
minimum_eculidean_distance=min(min(distance_matrix));
disp("Minimum distsnce be")
minimum_eculidean_distance
for i=1:height(distance_matrix)
    for j=1:length(distance_matrix)
        if distance_matrix(i,j)==minimum_eculidean_distance
            I=i;
            J=j;
            sub_table5=sub_table4(sub_table4.distance_matrix(i,j))
        
        end
    end
end
sub_table5
disp('ThankYou')