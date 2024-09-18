function [E] = NPE_my_approx(X, k)
[~,N] = size(X);
% X = X - repmat(mean(X,2), [1,N]);

X2 = sum(X.*X, 1); % 1*N
dist = repmat(X2, [N,1]) + repmat(X2', [1,N]) - 2.*X'*X;
[~, idx_n] = sort(dist); % dist的每列升序排. idx: N*N
neighbors = idx_n(2:k+1,:); % k*N
%% 构造W
W = zeros([N,N]);
tol = 1e-12;
for j=1:N
    Xj = repmat(X(:,j), [1,k]) - X(:, neighbors(:,j)); % D*k
    invG = inv( Xj'*Xj + tol*sum(sum(Xj.^2)).*eye(k) ); % k*k, 当K>D时, 警告：矩阵接近奇异值，或者缩放错误。结果可能不准确。
    w = sum(invG, 2) ./ sum(sum(invG)); % k*1
    W(neighbors(:,j), j) = w;
end

%% trick
% M = (diag(ones([N,1]))-W)*(diag(ones([N,1]))-W)'; % 
% % M = W*W' - W - W';
% % [U,S,V] = mySVD(X); % X:D*N  U:D*r  S:r*r  V:N*r
% [U,S,V] = svd(X,'econ'); % X:D*N  U:D*N  S:N*N  V:N*N
% T = V'*M*V; 
% T = max(T,T');
% [Evec, Eval] = eig(T); % V'*M*V = Evec * Eval * Evec': r*r
% Eval = diag(Eval);
% [~,idx_e] = sort(Eval); % 升序排
% Evec = Evec(:,idx_e); % Eval = Eval(idx_e); 
% % E = U*diag(1./diag(full(S)))*Evec;
% E = U*full(S)*Evec; % D*r
% for i=1:size(E,2)
%     E(:,i) = E(:,i)./sqrt(sum(E(:,i).*E(:,i)));
% end
%% trick-2
% % [U,S,V] = mySVD(X); % X:D*N  U:D*r  S:r*r  V:N*r
% [U,S,V] = svd(X,'econ'); % X:D*N  U:D*N  S:N*N  V:N*N  (r=N)
% S = diag(S); % N*1
% T = (V')*(diag(ones([N,1])) - W);
% T = T*T'; % r*r
% [Evec, Eval] = eig(T);
% Eval = diag(Eval);
% [~,idx_e] = sort(Eval); % 升序
% Evec = Evec(:,idx_e); % r*r
% % Eval = Eval(idx_e);
% E = U*diag(1./S)*Evec;
%% trick2 改成U:D*D版本
% [U,S,~] = svd(X); % X:D*N  U:D*D  S:D*N  V:N*N
[U,S,V] = svd(X, 'econ'); % X:D*N  U:D*N  S:N*N  V:N*N
s = diag(S); % N*1
% invS = [diag(1./s) zeros([N,D-N])]; % N*D
invS = diag(1./s); % N*N
T = V'*(eye(N)-W); % N*N  invS*(U')*X = V'
T = T*T'; % N*N
[Evec, Eval] = eig(T); 
Eval = diag(Eval); % N*1
[~,idx_e] = sort(Eval); % 升序
Evec = Evec(:,idx_e); % N*N
E = U*(invS')*Evec;

%% 严格遵循文章(不计时间代价)
% [U,~,~] = mySVD(X); % X:D*N  U:D*r  S:r*r  V:N*r
% X = U'*X; % D*N -> r*N
% T = (X*X')\(X*M*X'); % r*r
% [E, Eval] = eig(T);
% Eval = diag(Eval); % D*D -> D*1
% [~,idx_e] = sort(Eval); % 升序排
% E = E(:,idx_e); %Eval = Eval(idx_e); 

% [Evec, Eval] = eig(X*M*X',X*X'); % Evec:D*D
% Eval = diag(Eval); % D*D -> D*1
% [~,idx_e] = sort(Eval); % 升序排
% Evec = Evec(:,idx_e); %Eval = Eval(idx_e); 
% Y = Evec(:,1:d)'*X; % Y:(d*D)*(D*N) = d*N
return;

