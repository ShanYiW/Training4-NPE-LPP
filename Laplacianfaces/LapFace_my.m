function [Wpca,Wlpp] = LapFace_my(X, k, t)
% X: D*N, D: dimension, N: samples (D>>N)
% d: target dimension
% K: number of nearest neighbors
% E: D*r
N = size(X,2); % 样本数
mX = mean(X,2); % D*1
X = X - repmat(mX, [1,N]);
[Wpca] = PCA_DR(X, 0.995); % Wpca: D*r  Y=Wpca'*X: r*N
r = size(Wpca,2); % 列数
X = Wpca'*X; % D*N -> r*N  r<D
%% step 1 构造邻接矩阵
X2 = sum(X.*X, 1); % 1*N, [||X_1||2^2 ... ||X_N||2^2 ]
dist = repmat(X2, [N,1]) + repmat(X2', [1,N]) - 2.*X'*X; % N*N
% dist_{ij}: 样本Xi与Xj的欧氏距离, i,j \in {1,2,..., N}
% k近邻(取并集) 构造邻接矩阵
[~, nei_idx] = sort(dist);
nei_idx = nei_idx(2:k+1, :); % k*N
Adj = false([N,N]);
for j = 1:N
    Adj(nei_idx(:,j), j) = true;
    Adj(j, nei_idx(:,j)) = true;
end
% 构造权重矩阵
W = zeros([N,N]); 
W(Adj) = exp(-1.*dist(Adj)./(4*t)); % ||xi-xj||2^2 -> exp( -||xi-xj||2^2/(4t) )
%% 特征值分解
D_vec = sum(W, 2); % N*1
% XD_12 = X*diag(sqrt(D_vec)); % r*N
% XDXt = XD_12*XD_12';
% XLXt = X*(diag(D_vec) - W)*(X'); % r*r
% XLXt = max(XLXt, XLXt');
% [Wlpp, Eval] = eig(XLXt, XDXt); % Wlpp:r*r  Eval:r*r
% Eval = diag(Eval); % r*1
% [~, idx_e] = sort(Eval); % 升序排序
% Wlpp = Wlpp(:,idx_e); % Eval = Eval(idx_e);
% W = Wpca*Wlpp; % D*r * r*r = D*r
%%
D_12 = sqrt(D_vec); % N*1
[U,S,V] = svd(X, 'econ'); % X:r*N  U,S:r*r  V:N*r (N>>r)
M = eye(r) - V'*diag(1./D_12)*W*diag(1./D_12)*V;
M = max(M,M');
[Evec, Eval] = eig(M);
Eval = diag(Eval); [~,idx] = sort(Eval); Evec = Evec(:,idx);
invS = diag(1./diag(S));
Wlpp = U*invS*Evec; % r*r
% Y = Wlpp'*X*D_12; % 若svd(X), 则不*D_12. 若svd(XD_12), 则*D_12.
return;
