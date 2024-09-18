function [Y] = LPP_my(X, k, t, d)
% X: D*N, D: dimension, N: samples
% d: target dimension
% K: number of nearest neighbors
% Y: d*N
[D,N] = size(X);
X = X - repmat(mean(X,2), [1,N]); % 加上后 准确率提升15%
%% step 1 构造邻接矩阵
X2 = sum(X.*X, 1); % 1*N, [||X_1||2^2 ... ||X_N||2^2 ]
dist = repmat(X2, [N,1]) + repmat(X2', [1,N]) - 2.*X'*X; 
% dist_{ij}: 样本Xi与Xj的欧氏距离, i,j \in {1,2,..., N}
% k近邻(取并集) 构造邻接矩阵
[~, nei_idx] = sort(dist);
nei_idx = nei_idx(2:k+1, :); % k*N
Adj = false([N,N]);
for j = 1:N
    Adj(nei_idx(:,j), j) = true;
    Adj(j, nei_idx(:,j)) = true;
end
% num_of_neighbors = sum(Adj, 2); % N*1

% 构造权重矩阵
W = zeros([N,N]); 
W(Adj) = exp(-1.*dist(Adj)./(4*t)); % ||xi-xj||2^2 -> exp( -||xi-xj||2^2/(4t) )

D_vec = sum(W, 2); % N*1
%% 特征值分解
% XD = X*diag(sqrt(D_vec));
% [U,S,V] = svd(XD, 'econ'); % XD:D*N  U:D*D  S:D*D  V:N*D (D<N)
[U,S,V] = svd(X, 'econ'); % X:D*N  U:D*D  S:D*D  V:N*D (D<N)
S = diag(S);
[U,S,V] = CutonRatio(U,S,V, 0.98); % U:D*D->D*r  S:D*1->r*1  V:N*D->N*r 准确率提升15%
invS = diag(1./S); % r*r

M = eye(length(S)) - V'*diag(1./sqrt(D_vec))*W*diag(1./sqrt(D_vec))*V; % 机器误差更小
% M = eye(D) - V'*diag(1./sqrt(D_vec))*W*diag(1./sqrt(D_vec))*V; % 机器误差更小
M = max(M,M');
[Evec, Eval] = eig(M);
Eval = diag(Eval);
[~,idx_e] = sort(Eval);
Evec = Evec(:,idx_e); % Eval = Eval(idx_e);
E = U*(invS')*Evec;
% for i=1:size(E,2)
%     E(:,i) = E(:,i)./sqrt(sum(E(:,i).*E(:,i)));
% end
% Y = E(:,1:d)'*X*diag(sqrt(D_vec)); % d*N
Y = E(:,1:d)'*X; % d*N
return;
