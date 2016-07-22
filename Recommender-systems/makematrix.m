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



%% Create a matrix of size num_p by num_m from triplets {user_id, movie_id, rating_id}  

load train

num_m = 14726;
num_p = 223970;
count = sparse(num_p,num_m); %for Netflida data, use sparse matrix instead. 

for mm=1:num_m
 ff= find(M(:,2)==mm);
 fprintf(1, '\n %d / %d \t  \n', mm,num_m);
 count(M(ff,1),mm) = M(ff,3);
end 

save makematrix count
