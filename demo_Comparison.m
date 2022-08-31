
 clear
 clc

 %Add the path to dataset
 load('C:\Users\abray\Documents\UDA-TFL\data\WebcamDecaf_vs_DSLRDecaf (W-D).mat')

% %image dimensions
 div1=64;
 div2=64;
 dsName='WebcamDecaf DSLRDecaf';

total_results=cell(5,1);

%reshape the data
n=size(X_src,2);
TX = reshape(X_src,[div1,div2,n]);

n=size(X_tar,2);
TS = reshape(X_tar,[div1,div2,n]);

testQ=97;
maxK=1;


Y_tar_pseudo=[];

startP=50;
startlambda=0.1;

results_optm_lambda=[];
maxiter=20;
maxlambda=1;

for lambda=startlambda:0.1:maxlambda
stoploop=0;   
for P=startP:100:1000
	
for maxK=1:maxiter
 
[tUs, odrIdx, TXmean, Wgt]  =TFL(TX,Y_src,TS,Y_tar,testQ,maxK,Y_tar_pseudo,lambda);

if(P>size(tUs{1},1)*size(tUs{2},1))
P=size(tUs{1},1)*size(tUs{2},1)
stoploop=1;
end
 
acc = classify_output(TX,Y_src,TS,Y_tar,TXmean,tUs,odrIdx,P,Wgt,'knn')
fprintf('UDA-TFL-1NN=%0.4f\n',acc);
  
%end
results_optm_lambda=[results_optm_lambda;[lambda P maxK acc]];
    
	if(stoploop)
	 break
    end
end
end
end
total_results{5,:}=results_optm_lambda;

%save(strcat('Exp/moreK_results_',dsName,'.mat'),'total_results');
