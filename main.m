%% Initialization
clear ; close all; clc

data2 = importdata('data/arrowhead/time_series_data/arrowhead_train');
data = importdata('data/Coffee/Coffee_TRAIN');

%%%%%%%%%%%%%%%%%%%%%%%%% REQUIRED VARIABLES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

T = data(:,2:end);
[I, Q] = size(T);								% I (number of training samples), Q (length of each sample)

Y = data(:,1);
Yu = unique(Y);
C = size(Yu)(1);
												% Yb = ones(I, C);
for c = 1:C
	Yb(:,c) = (Y==c);
end

lambda = 0.01;									% regularization parameter 
eta = 0.01; 									% learning rate 
R = 3; 											% scales of shapelet lengths	
K = 3 %0.05*Q;										% number of shapelets
Lmin = 0.175*Q;									% minimum length of shapelets				

maxIter = 5;									% max iteration

alpha = -1; %% should be -100					% controls the precision of the function & 
												% the soft minimum -> the true min for alpha -> inf

%%%%%%%%%%%%%%%%%%%%%%%%% SHAPELETS INITIALIZATION %%%%%%%%%%%%%%%%%%%%%%%%%%

W = zeros(C, R, K);
W0 = zeros(C,1);

S{1,1} = zeros(K, Lmin);

for r = 2:R
	S{1,r} = zeros(K, r*Lmin);
end

function Jr = J(r, q, lmin)
	Jr = q - r*lmin + 1;
end
M_Matrix = [];
for iteration = 1:maxIter
	for i = 1:I
		% {Pre-compute Terms}
		for r = 1:R
			for k = 1:K
				for j = 1:J(r, Q, Lmin)
					d = sum((T(i, j:j+r*Lmin-1) - S{1,r}(k,:)).^2) / (r*Lmin);
					D{r,i}(k,j) = d;
					E{r,i}(k,j) = exp(alpha*d);

				endfor
				p = sum(E{r,i}(k,1:J(r, Q, Lmin)));
				P{r,i}(k) = p;
				M{r,i}(k) = sum(D{r,i}(k,1:J(r, Q, Lmin)) * E{r,i}(k,1:J(r, Q, Lmin))') / (1/p);
			endfor %'
		endfor

		for c = 1:C
			z = 0;
			for _r = 1:R
				z += sum(M{_r,i}*W(c,_r));
			endfor 
			Yh(i,c) = sigmoid(z);
			V(i,c) = Yb(i,c) - Yh(i,c);
		endfor

		% {Learn Shapelets and Classification Weights}
		for c = 1:C
			for r = 1:R
				for k = 1:K
					W(c,r,k) += eta*(V(i,c)*M{r,i}(k) - (2*lambda)*W(c,r,k)/(I*C));
					for j = 1:J(r, Q, Lmin)
						Phi{r,i}(k,j) = 2*E{r,i}(k,j)*(1 + alpha*(D{r,i}(k,j) - M{r,i}(k))) / (r*Lmin*P{r,i}(k));
						for l = 1:r*Lmin
							S{1,r}(k,l) += eta*V(i,c)*Phi{r,i}(k,j)*(S{1,r}(k,l) - T(i,j+l-1)).*W(c,r,k);
						endfor
					endfor
				endfor
			endfor
			W0(c,1) += eta*V(i,c);
		endfor
	
	endfor
endfor
