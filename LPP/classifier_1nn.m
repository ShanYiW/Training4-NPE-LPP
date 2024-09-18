function accuracy = classifier_1nn(Y, gnd)
N = size(Y,2);
Y2 = sum(Y.*Y, 1); % 1*N
dist = repmat(Y2, [N,1]) + repmat(Y2', [1,N]) - 2.*Y'*Y; % N*N
[~, nei_idx] = sort(dist);
class_pred = gnd( nei_idx(2,:) );

accuracy = sum(class_pred == gnd) / N;
return;