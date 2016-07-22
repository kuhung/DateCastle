function [pred_out] = pred(w1_M1_sample,w1_P1_sample,N,mean_rating);

%%% Make predicitions on the validation data

 aa_p   = double(N(:,1));
 aa_m   = double(N(:,2));
 rating = double(N(:,3));

 pred_out = sum(w1_M1_sample(aa_m,:).*w1_P1_sample(aa_p,:),2) + mean_rating;
 ff = find(pred_out>5); pred_out(ff)=5;
 ff = find(pred_out<1); pred_out(ff)=1;


 
