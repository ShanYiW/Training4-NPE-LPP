function pred = classifier_knn(y, Y, gnd, k, N_class)
% Input:
% y: d*1. 待分类的样本
% Y: d*N. 用于分类的数据
% gnd: N*1. Y的N个样本的类别
% k: 近邻数
% N_class: 类别总数
% Output:
% pred: 预测的类别

N = size(Y,2);
dist = sum((repmat(y,[1,N]) - Y).^2); % 1*N. 测试样本y与N个样本的距离
[~, nei_idx] = sort(dist); % 升序排序
nei_set = gnd(nei_idx(1:k)); % 1*k, 最近k个样本的类别
num_per_class = zeros([N_class,1]); % 计数.
for i=1:k % 对于每个最近邻
    num_per_class(nei_set(i)) = num_per_class(nei_set(i)) + 1/(dist(nei_idx(i))+eps); % 1
end

pred = 0; num_this_class = -1;
for i=1:N_class % 检查每个类的最近邻数量, 选出邻居最多的类 赋值pred
    if num_per_class(i)>num_this_class
        num_this_class = num_per_class(i);
        pred = i;
    end
end
return;