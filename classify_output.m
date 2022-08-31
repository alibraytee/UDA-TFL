function acc = classify_output(TX,Y_src,TS,Y_tar,TXmean,tUs,odrIdx,P,wgts,learner)

 
N=ndims(TX)-1;%The order of samples.
    IsTX=size(TX);
    Is=IsTX(1:N);%The dimensions of the tensor
    numSpl=IsTX(N+1);%Number of samples

    fea3Dctr=TX-repmat(TXmean,[ones(1,N), numSpl]);%Centering
    newfea = ttm(tensor(fea3Dctr),tUs,1:N);%MPCA projection
    %Vectorization of the tensorial feature
    newfeaDim=size(newfea,1)*size(newfea,2);
    train_mat=reshape(newfea.data,newfeaDim,numSpl)';%Note: Transposed
    train_mat=real(train_mat);
	
	train_mat=train_mat(:,odrIdx(1:P));

%    selfea=newfea(:,odrIdx(1:P));%Select the first "P" sorted features
%       %standard classifier (e.g., nearest neighbor classifier), you may 
    numSpl=size(TS,N+1);
    fea3Dctr=TS-repmat(TXmean,[ones(1,N), numSpl]);%Centering
    newfea = ttm(tensor(fea3Dctr),tUs,1:N);%MPCA projection
    %Vectorization of the tensorial feature
    newfeaDim=size(newfea,1)*size(newfea,2);
    test_mat=reshape(newfea.data,newfeaDim,numSpl)';%Note: Transposed
test_mat=real(test_mat);
	test_mat=test_mat(:,odrIdx(1:P));
	
    train_mat=zscore(train_mat);
     test_mat=zscore(test_mat);
     
    
    if(learner=='knn')
        knn_model= fitcknn(train_mat,Y_src,'NumNeighbors',1);

 
        Y_tar_pseudo = knn_model.predict(test_mat);
        
    else
        svm_model0=fitcecoc(train_mat,Y_src);
        Y_tar_pseudo = predict(svm_model0,test_mat);
    end
 
        acc = length(find(Y_tar_pseudo==Y_tar))/length(Y_tar_pseudo); 
      
            fprintf(['UDA-TFL-',learner,'=%0.4f\n'],acc);
end

