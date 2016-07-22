% Version 1.000
%
% Code provided by Ruslan Salakhutdinov
%
% Permission is granted for anyone to copy, use, modify, or distribute this
% program and accompanying programs and documents for any purpose, provided
% this copyright notice is retained and prominently displayed, along with
% a note saying that the original programs are available from our
% web page.
% The programs and documents are distributed without any warranty, express or
% implied.  As the programs were written for research purposes only, they have
% not been tested to the degree that would be advisable in any important
% application.  All use of these programs is entirely at the user's own risk.

rand('state',0); 
randn('state',0); 

if restart==1 
  restart=0;
  epsilon=50; % Learning rate 学习比率
  lambda  = 0.01; % Regularization parameter 正则化参数
  momentum=0.8; 

  epoch=1; 
  maxepoch=50; % 最大阶数 

  load train % Triplets: {user_id, movie_id, rating} 三个一组
  mean_rating = mean( M (:,3)); % 平均评分
  
  pairs_tr = length(M); % training data 训练数据

  numbatches= 210; % Number of batches 批数
  num_m = 14726;  % Number of movies 电影数
  num_p = 223970;  % Number of users 用户数
  num_feat = 10; % Rank 10 decomposition 10次分解？

  w1_M1     = 0.1*randn(num_m, num_feat); % Movie feature vectors 电影特征矩阵
  w1_P1     = 0.1*randn(num_p, num_feat); % User feature vecators 用户特征矩阵
  w1_M1_inc = zeros(num_m, num_feat); % 初始化
  w1_P1_inc = zeros(num_p, num_feat); % 初始化

end


for epoch = epoch:maxepoch % 迭代开始
  rr = randperm(pairs_tr); % 随机置换向量
  M = M(rr,:); % 训练集随机化
  clear rr 

  for batch = 1:numbatches
    fprintf(1,'epoch %d batch %d \r',epoch,batch);
    N=157987; % number training triplets per batch 每一批的训练集样本数

    aa_p   = double(M((batch-1)*N+1:batch*N,1));
    aa_m   = double(M((batch-1)*N+1:batch*N,2));
    rating = double(M((batch-1)*N+1:batch*N,3));

    rating = rating-mean_rating; % Default prediction is the mean rating. 默认为平均评分

    %%%%%%%%%%%%%% Compute Predictions 计算预测 %%%%%%%%%%%%%%%%%
    pred_out = sum(w1_M1(aa_m,:).*w1_P1(aa_p,:),2); % 2代表矩阵相乘后行求和
    f = sum( (pred_out - rating).^2 + ...
        0.5*lambda*( sum( (w1_M1(aa_m,:).^2 + w1_P1(aa_p,:).^2),2)));

    %%%%%%%%%%%%%% Compute Gradients 计算梯度 %%%%%%%%%%%%%%%%%%%
    IO = repmat(2*(pred_out - rating),1,num_feat); % repmat 矩阵重复函数
    Ix_m=IO.*w1_P1(aa_p,:) + lambda*w1_M1(aa_m,:);
    Ix_p=IO.*w1_M1(aa_m,:) + lambda*w1_P1(aa_p,:);

    dw1_M1 = zeros(num_m,num_feat);
    dw1_P1 = zeros(num_p,num_feat);

    for ii=1:N
      dw1_M1(aa_m(ii),:) =  dw1_M1(aa_m(ii),:) +  Ix_m(ii,:);
      dw1_P1(aa_p(ii),:) =  dw1_P1(aa_p(ii),:) +  Ix_p(ii,:);
    end

    %%%% Update movie and user features 更新电影和用户特征 %%%%%%%%%%%

    w1_M1_inc = momentum*w1_M1_inc + epsilon*dw1_M1/N;
    w1_M1 =  w1_M1 - w1_M1_inc;

    w1_P1_inc = momentum*w1_P1_inc + epsilon*dw1_P1/N;
    w1_P1 =  w1_P1 - w1_P1_inc;
  end 

  %%%%%%%%%%%%%% Compute Predictions after Paramete Updates 字段更新后的再一次预测 %%%%%%%%%%%%%%%%%
  pred_out = sum(w1_M1(aa_m,:).*w1_P1(aa_p,:),2);
  f_s = sum( (pred_out - rating).^2 + ...
        0.5*lambda*( sum( (w1_M1(aa_m,:).^2 + w1_P1(aa_p,:).^2),2)));
  err_train(epoch) = sqrt(f_s/N);

  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  %%% Compute predictions on the test set 测试集上的预测 %%%%%%%%%%%%%%%%%%%%%% 
  load test
  
  pairs_pr = length(test); 
  NN=pairs_pr;

  aa_p = double(test(:,1));
  aa_m = double(test(:,2));
  %rating = double(test(:,3));

  pred_out = sum(w1_M1(aa_m,:).*w1_P1(aa_p,:),2) + mean_rating;
  ff = find(pred_out>5); pred_out(ff)=5; % Clip predictions 预测修剪
  ff = find(pred_out<1); pred_out(ff)=1;

  fprintf(1, 'epoch %4i batch %4i Training RMSE %6.4f  \n', ...
              epoch, batch, err_train(epoch));
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

  if (rem(epoch,10))==0
     save pmf_weight w1_M1 w1_P1
  end

end 



