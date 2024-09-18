clear;
% warning off;
%% 载入 MNIST 数据集
load('2k2k.mat'); % fea: 4000*784, gnd: 4000*1, trainIdx,testIdx: 4000*1
% 分离训练集、测试集
train_set = fea(trainIdx,:); % 训练集2000*784
gnd = gnd(trainIdx); % 2000*1, 每个样本的类别
X = train_set';
% test_set = fea(testIdx,:); % 测试集2000*784
% gnd = gnd(testIdx); % 2000*1
% X = test_set'; % -> 784*2000, 784特征数, 2000样本数
% X = X./255;
classes = unique(gnd); % [0; 1; ...; 9]
n_class = length(classes); % 类别数目
colors = ["r","g","b","c","m","y","k","#D95319","#7E2F8E","#0072BD"];
%% 
% lb_t   =  300000;
% ub_t   = 1000001;
% step_t =  700000;
t = 3e5; % 
% k = 5;
lb_k = 3;
ub_k = 15;
step_k=1;

for k = [5:10:46]
[Y] = LPP_my(X, k, t, 2);
% Y = E(:,1:2)'*X;
acc = classifier_1nn(Y(1:2,:),gnd);
figure('Name', ['k=',num2str(k),', acc=',num2str(acc)]);
for i = 1:n_class
    sample_this_class = Y(1:2, find(gnd==classes(i)));
    scatter(sample_this_class(1,:), sample_this_class(2,:), 'o', "MarkerEdgeColor", colors(i)); hold on;
end
xlabel('$\mathbf{Y}^{(1)}$', 'Interpreter', 'latex');
ylabel('$\mathbf{Y}^{(2)}$', 'Interpreter', 'latex');
legend('0','1','2','3','4','5','6','7','8','9', 'Location', 'southeast'); % 'northwest'
axis('equal'); hold off;

end

