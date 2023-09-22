%%Here we the read the data%%
data=xlsread('D:\MFO1\Record.xlsx');
size(data);
%%Here we transpose our real df%%
data=transpose(data);
disp("size of the original dataframe be")
size(data)

%%Now we take the user input for particular iteration%%
prompt="Enter the iteration";
iteration=input(prompt);
disp(iteration)

%%we store the data from df for the particular iteration into andother df
data1=data(:,iteration);
disp("Size of the dataframe for particular iteration")
size(data1)

%%here we store the index into a list
Index=[];
for i=1:length(data1)
    Index=[Index i];
end
Index=transpose(Index);
disp("size of the index of data")
size(Index)

%%we draw the scatter plot of new dataframe%%
plot(Index,data1,'r*')
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
epsilon=clusterDBSCAN.estimateEpsilon(data1,minpts,maxpts);
disp("Epsilon value be")
epsilon


%%%%%%%%%%%%%%%%%%%After getting the value of epsilon we impelemt the
%%%%%%%%%%%%%%%%%%%dbscan alogrithm%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
idx=dbscan(data1,epsilon,minpts);
disp("Size of idx be")
size(idx)
unique_idx=unique(idx)
disp("Length of unique idx value be")
disp(length(unique_idx))

%here we add the labels into data frame
T=table(Index,data1,idx);
T(1:5,:)

%%%Now we find the centroid of each cluster.%%%
n1=[];
n2=[];
centroid=[];
for i=1:length(unique_idx)
    n1=[n1 unique_idx(i)];
    
    %print the subtable for each cluster_labels%
    sub_table=T(T.idx==unique_idx(i),:);
    n2=[n2 height(sub_table)];
    
    %find the mean of the each cluster
    mean_value=mean(sub_table.data1);
    centroid=[centroid mean_value];
end
 %display the value of n1, n2 and centroid and assign in a single table%
 Cluster_Labels=transpose(n1);
 Size_of_each_cluster=transpose(n2);
 centroid=transpose(centroid);
 Cluster_Labels;
 Size_of_each_cluster;
 centroid;
dataframe=table(Cluster_Labels,Size_of_each_cluster,centroid);
dataframe


%%%%%%%%%%%For Outlier Work%%%%%%%%%%%%%%

%(i) Find the centroid of outlier and store into excel%
if dataframe.Cluster_Labels==-1
    outlier_centroid=dataframe.centroid(dataframe.Cluster_Labels==-1);
    disp("Outlier centroid be")
    outlier_centroid
else
    disp('No Outlier Present')
end
%(ii) Find the indexes of Outliers and store into the excel
if dataframe.Cluster_Labels==-1
    for i=1:length(unique_idx)
        if unique_idx(i)==-1
            sub_table1=T(T.idx==-1,:);    
        end
    end
    sub_table1
    disp("length of index be")
    disp(height(sub_table1))
else
    disp('No Outlier Present')
end
    
%%%%%%%%%%%%%%%%%%%%%For Non Outlier Work%%%%%%%%%%%%%%%%%%%%%%

%(i) Find the maximum value of the centroid of the clusters(except outlier cluster)
sub_table2=dataframe(dataframe.Cluster_Labels~=-1,:);
sub_table2

maximum_centroid_value=max(sub_table2.centroid);
disp("Maximum centroid be")
maximum_centroid_value
maximum_table=sub_table2(sub_table2.centroid==maximum_centroid_value,:);
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
eculidean_distances=[];
for i=1:length(sub_table3.data1)
    distance=pdist2(maximum_centroid_value,sub_table3.data1(i));
    eculidean_distances=[eculidean_distances distance];
end
%eculidean_distances
eculidean_distance=transpose(eculidean_distances);
eculidean_distance;
disp("length of eculidean distance vector")
length(eculidean_distance)

sub_table4=table(sub_table3,eculidean_distance);

%Now we find the our objective in this question%
minimum_eculidean_distance=min(eculidean_distance);
disp("Minimum distsnce be")
minimum_eculidean_distance
sub_table5=sub_table4(sub_table4.eculidean_distance==minimum_eculidean_distance,:);
sub_table5