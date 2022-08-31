function [tUs, odrIdx, TXmean, Wgt]  = TFL(X_src,Y_src,X_tar,Y_tar,testQ,maxK,Y_tar_pseudo,lambda)
%Using MPCA and MMD
% MPCA code is used with MMD: Multilinear {P}rincipal {C}omponent {A}nalysis of {T}ensor {O}bjects
%UDA-TFL: The low dimensional tensor samples minimize the difference between source
%and domain distributions using ALS optimization
%We cited the paper in manuscript
% %[Prototype]%
 
% %[Inputs]%:
%    TX: the input training data in tensorial representation, the last mode
%        is the sample mode. For Nth-order tensor data, TX is of 
%        (N+1)th-order with the (N+1)-mode to be the sample mode.
%        E.g., 30x20x10x100 for 100 samples of size 30x20x10
%        If your training data is too big, resulting in the "out of memory"
%        error, you could work around this problem by reading samples one 
%        by one from the harddisk, or you could email me for help.
%
%    gndTX: the ground truth class labels (1,2,3,...) for the training data
%           E.g., a 100x1 vector if there are 100 samples
%           If the class label is not available (unsupervised learning),
%           please set gndTX=-1;
%
%    testQ: the percentage of variation kept in each mode, suggested value
%           is 97, and you can try other values, e.g., from 95 to 100, to
%           see whether better performance can be obtained.
%
%    maxK: the maximum number of iterations, suggested value is 1, and you 
%          can try a larger value if computational time is not a concern.
%
% %[Outputs]%:
%    tUs: the multilinear projection, consiting of N
%         projection matrices, one for each mode
%
%    odrIdx: the ordering index of projected features in decreasing  
%            variance (if unsupervised) or discriminality (if supervised)  
%            for vectorizing the projected tensorial features
%
%    TXmean: the mean of the input training samples TX
%
%    Wgt: the weight tensor for use in modified distance measures. Please
%         refer to Section IV.B and IV.C of the paper.
%
 
%TX: (N+1)-dimensional tensor of Tensor Sample Dimension x NumSamples
TX=cat(ndims(X_src),X_src,X_tar);
gndTX=[Y_src;Y_tar];

N=ndims(TX)-1;%The order of samples.
IsTX=size(TX);
Is=IsTX(1:N);%The dimensions of the tensor
numSpl=IsTX(N+1);%Number of samples


%%%%%%%%%%%%%%Generate MMD0
%%%%added by AB
n = size(TX,ndims(TX));
ns = size(X_src,ndims(X_src));
nt = size(X_tar,ndims(X_tar));

e = [1/ns*ones(ns,1);1/nt*ones(nt,1)];
%correct one
%e = [1/ns*ones(ns,1);-1/nt*ones(nt,1)];
C = length(unique(Y_src));
%%% M0
M = e * C; %  * e' * C;  %multiply C for better normalization

%%% Mc
	NN = 0;
	if ~isempty(Y_tar_pseudo) && length(Y_tar_pseudo)==nt
		for c = reshape(unique(Y_src),1,C)
			e = zeros(n,1);
			e(Y_src==c) = 1 / length(find(Y_src==c));
			e(ns+find(Y_tar_pseudo==c)) = -1 / length(find(Y_tar_pseudo==c));
			e(isinf(e)) = 0;
			%N = N + e*e';
            NN = NN + e;
		end
	end

M = M + NN;
M = M / norm(M,'fro');

switch N
    case 2
        M = repmat(reshape(M,1,1,[]),size(TX,1),size(TX,2));
    case 3
        M = repmat(reshape(M,1,1,[]),size(TX,1),size(TX,2),size(TX,3));
    case 4
        M = repmat(reshape(M,1,1,[]),size(TX,1),size(TX,2),size(TX,3),size(TX,4));
    otherwise
        error('Order N not supported. Please modify the code here or email hplu@ieee.org for help.')
end
        

%%%%% Centering matrix H
%H = eye(n) - 1/n * ones(n,n);
    
%%%%end added by AB
%%%%%%%%%%%%%Zero-Mean%%%%%%%%%%
%step 1 in Fig3 algo in original paper 
TXmean=mean(TX,N+1);%The mean
TX=TX-repmat(TXmean,[ones(1,N), numSpl]);%Centering

%The full projection for initialization
Qs=ones(N,1)*testQ;
Us=cell(N,1);
tUs=cell(N,1);
Lmds=cell(N,1);
for n=1:N
    In=Is(n);Phi=zeros(In,In);PhiHn=zeros(In,In);
    for m=1:numSpl
        switch N
            case 2
                Xm=TX(:,:,m);
                Mm=M(:,:,m);
            case 3
                Xm=TX(:,:,:,m);
                Mm=M(:,:,:,m);
            case 4
                Xm=TX(:,:,:,:,m);
                Mm=M(:,:,:,:,m);
            otherwise
                error('Order N not supported. Please modify the code here or email hplu@ieee.org for help.')
        end
        tX=tensor(Xm);
        tXn=tenmat(tX,n);
        Xn=tXn.data;
       
     
        %Phi=Phi+Xn*Xn';
        
        Phi=Phi+Xn*Mm*Xn' + lambda*eye(size(Xn,1));
        PhiHn=PhiHn +Xn*TXmean*Xn';
    end
    %step 2 in Fig3 algo in original paper
    option=struct('disp',0);
     kk=size(Phi,1)-1;
     [Un,Lmdn]=eigs(Phi,PhiHn,kk,'lm',option);
    %[Un,Lmdn]=eigs(Phi,PhiHn,'lm',option);
   
   % [Un,Lmdn]=eig(Phi);
    
    Lmd=diag(Lmdn);
    
       Us{n}=Un;
      tUs{n}=Us{n}';
       Lmds{n}=Lmd;

   %[stLmd,stIdx]=sort(Lmd,'descend');
   %Us{n}=Un(:,stIdx);
   %tUs{n}=Us{n}';
   %Lmds{n}=Lmd(stIdx);
end

%Cumulative distribution of eigenvalues
%this code is needed to check how many principal compoenets are needed to
%cover the variance of Qs (e.g 97%)
cums=cell(N,1);
for n=1:N
    In=length(Lmds{n});
    cumLmds=zeros(In,1);
    Lmd=Lmds{n};
    cumLmds(1)=Lmd(1);
    for in=2:In
        cumLmds(in)=cumLmds(in-1)+Lmd(in);
    end
    cumLmds=cumLmds./sum(Lmd);
    cums{n}=cumLmds;
end


%local optimization ALS approac
if maxK>0
    tPs=cell(N,1);
    pUs=cell(N,1);
    %%%%%%%%%%%%%Determine Rn, the dimension of projected space%%%%
    %%%keep PCs based on Qs variation
    for n=1:N
        cum=cums{n};
        idxs=find(cum>=Qs(n)/100);
        Ps(n)=idxs(1);
        tUn=tUs{n};
        tPn=tUn(1:Ps(n),:);
        tPs{n}=tPn;
    end
    %tPs holds the projected data based on survived PCs which over variance
    %limit
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    for iK=1:maxK
        for n=1:N
            In=Is(n);
            Phi=double(zeros(In,In));
            for m=1:numSpl
                switch N
                    case 2
                        Xm=TX(:,:,m);
                        Mm=M(:,:,m);
                    case 3
                        Xm=TX(:,:,:,m);
                        Mm=M(:,:,:,m);
                    case 4
                        Xm=TX(:,:,:,:,m);
                        Mm=M(:,:,:,:,m); 
                    otherwise
                        error('Order N not supported. Please modify the code here or email hplu@ieee.org for help.')
                end
                %tensor only data got from PCA using preserved PCs
                %%ttm Tensor times matrix for ktensor
                %multiplying here to project the original data using
                %coefficent matrix in tPs
                
                %step 3.1 in algo code in paper
                tX=ttm(tensor(Xm),tPs,-n);
                %tenmat Converting a tensor to a matrix and vice versa
                tXn=tenmat(tX,n);
                Xn=tXn.data;
                %step 3.2 forbenious form
                %Phi=Phi+Xn*Xn';
                Phi=Phi+Xn*Xn'*Mm + lambda*eye(size(Xn,1));
            end
            Pn=Ps(n);
            Phi=double(Phi);
            if Pn<In
                option=struct('disp',0);
                [pUs{n},pLmdn]=eigs(Phi,Pn,'lm',option);
                pLmds{n}=diag(pLmdn);
            else
                [pUn,pLmdn]=eig(Phi);
                pLmd=diag(pLmdn);
                %step 3.3 from algo pseudo code
                [stLmd,stIdx]=sort(pLmd,'descend');
                pUs{n}=pUn(:,stIdx(1:Pn));
                pLmds{n}=pLmd(stIdx(1:Pn));
            end
            tPs{n}=pUs{n}';
        end
    end
    Us=pUs;
    tUs=tPs;
    Lmds=pLmds;
    Is=Ps;
else
    if testQ<100
        error('At least one iteration is needed');
    end
end
%step 4 in algo the projected feature tensor in tUs
%Calculate the weight tensor Wgt
Wgt=zeros(Is);
switch N
    case 2
        for i1=1:Is(1)
            for i2=1:Is(2)
                Wgt(i1,i2)=sqrt(Lmds{1}(i1)*Lmds{2}(i2));
            end
        end
    case 3
        for i1=1:Is(1)
            for i2=1:Is(2)
                for i3=1:Is(3)
                    Wgt(i1,i2,i3)=sqrt(Lmds{1}(i1)*Lmds{2}(i2)*Lmds{3}(i3));
                end
            end
        end
    case 4
        for i1=1:Is(1)
            for i2=1:Is(2)
                for i3=1:Is(3)
                    for i4=1:Is(4)
                        Wgt(i1,i2,i3,i4)=sqrt(Lmds{1}(i1)*Lmds{2}(i2)*Lmds{3}(i3)*Lmds{4}(i4));
                    end
                end
            end
        end
    otherwise
        error('Order N not supported. Please modify the code here or email hplu@ieee.org for help.')
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% tUs projected features
%step 4 in code algo
Yps=ttm(tensor(TX),tUs,1:N);% projections of samples TX
vecDim=1;
for n=1:N, vecDim=vecDim*Is(n); end
vecYps=reshape(Yps.data,vecDim,numSpl); %vectorization of Yps
%%%%%%%%%%%%%%Now vecYps contains the feature vectors for training data

if max(gndTX)<0%%%%%%%%%%%%%%%%%%%%%%%%Sort by Variance%%%%%%%%%%%%%%%%%%%%
    disp("sort by variance")
    TVars=diag(vecYps*vecYps');
    [stTVars,odrIdx]=sort(TVars,'descend');
else%%%%%%%%%%%%%%%Sort according to Fisher's discriminality%%%%%%%%%%%%%%%
     %disp("sort by Fisher")
      
    classLabel = unique(gndTX);
    nClass = length(classLabel);%Number of classes
    ClsIdxs=cell(nClass);
    Ns=zeros(nClass,1);
    for i=1:nClass
        ClsIdxs{i}=find(gndTX==classLabel(i));
        Ns(i)=length(ClsIdxs{i});
    end
    Ymean=mean(vecYps,2);
    TSW=zeros(vecDim,1);
    TSB=zeros(vecDim,1);
    for i=1:nClass
        clsYp=vecYps(:,ClsIdxs{i});
        clsMean=mean(clsYp,2);
        FtrDiff=clsYp-repmat(clsMean,1,Ns(i));
        TSW=TSW+sum(FtrDiff.*FtrDiff,2);
        meanDiff=clsMean-Ymean;
        TSB=TSB+Ns(i)*meanDiff.*meanDiff;
    end
    FisherRatio=TSB./TSW;
    [stRatio,odrIdx]=sort(FisherRatio,'descend');
end