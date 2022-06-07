clear all
clc
addpath([pwd, '/my_tools']);
addpath([pwd, '/dataset']);
anchor_rate =0.7;% the anchor rate to all samples
% %%======MSRC=================%%
load  MSRC;  %Read database
gt = Y;    % the grount truth for testing
cls_num = length(unique(gt)); % the number of clusters
nV = length(X); %Get the number of views
disp('----------Data preprocessing----------')
for v = 1:nV
    a = max(X{v}(:));
    X{v} = double(X{v}./a); 
end  %Data preprocessing

fid=fopen('MSRC result.txt ','a');% build the txt file to save clustering results
lambda2_v=[100]; %Initialize tensor coefficients
lpp = [0.1];  %the p-value of tensor Schatten p-norm
k_np = 5;
n_sample = size(X{1},1); %Number of samples
M = fix( n_sample * anchor_rate); %Number of anchor points
dim = 20; %Dimension after projection
for k = 1:length(lambda2_v)
    lambda2= lambda2_v(k);
    for kkk=1:length(lpp)
        beta =ones(1,nV)';  % the defult weight_vector of tensor Schatten p-norm
        lp=lpp(kkk);
        betaf =1;
        %% ==================Training ================%%
        [Z,E,F,S,history1] = my_test1(X, cls_num, M, beta, lp, k_np,lambda2, betaf); 
        %% ================Obtain  Graph=============%%
        Final_S =sum(cat(3,S{:}),3)./nV;
        IterMax = 50;
        disp('----------Clustering----------');
        tic
        %% ==Obtain clustering results%%
        [Flabel,~,~,~,~,~] = coclustering_bipartite_fast1(Final_S, cls_num, IterMax);
        toc
        rs(:,1) = Flabel;
        %% =============Measure the performance=============%%
        myresult = ClusteringMeasure1(Y, rs(:,1));
        myresult
        %% ======================Record=====================%%
        fprintf(fid,'lambda2: %f ',lambda2);
        fprintf(fid,'lp: %f ',lp);
        fprintf(fid,'beta: %g %g %g  \n',beta);
        fprintf(fid,'%g  %g %g %g %g  %g %g \n ',myresult');
    end
end
fclose(fid);