clear;
% ORL = load('ORL.mat');
% ORL = double(ORL.ORL);
% ORL = ORL./255;
% [D,N] = size(ORL);
% %% 生成类别
% gnd = zeros([N,1]); % 类别(哪个人)标签
% for person = 1:N
%     gnd((person-1)*10+1:person*10) = person;
% end
load('ORL_32.mat');
ORL = fea'; clear fea;
[D,N] = size(ORL);
N_class = 40; N_per_class = 10;
%%
L = 2;
N_train = L*N_class; % 训练样本数
N_test = N-N_train;
trainset = zeros([D, N_train]); % 训练集
testset = zeros([D,N-N_train]); % 测试集
gnd_train = zeros([N_train,1]);
gnd_test = zeros([N_test,1]); % gnd_test(i)=第i个测试样本的类别

lb_d=10; ub_d=80;
len_d = ub_d-lb_d+1;
lb_k=10; ub_k=75; step_k=5; % L=2
% lb_k=10; ub_k=110; step_k=10; % L=3
% lb_k=10; ub_k=140; step_k=10; % L=4
% lb_k=5; ub_k=185; step_k=15; % L=5
len_k = (ub_k-lb_k)/step_k + 1;

Acc_dk = zeros([len_k, len_d]);
N_rnd = 20; % 重复次数

for k=lb_k:step_k:ub_k
acc_rnd_d = zeros([N_rnd, len_d]); % 
for rnd = 1:N_rnd
%% 生成训练集, 测试集
i_tr = 0; i_te = 0;
for j=1:N_class % 40个人
    train_idx = sort(randperm(N_per_class,L)); % 升序排序
    trainset(:,i_tr+1:i_tr+L) = ORL(:, [(j-1)*N_per_class + train_idx]); % 不能[(j-1)*N_per_class +train_idx]
    gnd_train(i_tr+1:i_tr+L) = gnd([(j-1)*N_per_class + train_idx]);
    i_tr = i_tr+L;
    
    test_idx = setdiff(1:N_per_class,train_idx); % 集合作差: 全集-train_idx
    testset(:,i_te+1:i_te+N_per_class-L) = ORL(:,[(j-1)*N_per_class + test_idx]);
    gnd_test(i_te+1:i_te+N_per_class-L) = gnd([(j-1)*N_per_class + test_idx]);
    i_te = i_te+N_per_class-L;
end
%% 运行 NPE, 输出E
% options.NeighborMode = 'KNN'; options.ReducedDim=d;
% [Evec, Eval] = NPE(options, trainset');
% Y_train = Evec(:,1:d)'*trainset;
% Y_test = Evec(:,1:d)'*testset;

[E] = NPE_my_approx(trainset, k); % E: D*D
%% 计算分类准确度
for d=lb_d:ub_d
Y_train = E(:,1:d)'*trainset;
Y_test  = E(:,1:d)'*testset;
acc = 0;
for j=1:N_test
    y = Y_test(:,j); % d*1
%     pred = classifier_knn(y, Y_train, gnd_train, 4, N_class);
    dist = sum((repmat(y,[1,N_train]) - Y_train).^2, 1);
    [~,idx] = sort(dist); % 距离 升序排
    pred = gnd_train(idx(1)); % 最近邻的类别 = j号测试样本的类别
    acc = acc + (pred==gnd_test(j));
end
acc = acc/N_test; % 
acc_rnd_d(rnd, d-lb_d+1) = acc; % 每次随机测试的准确度 <- 所有测试样本
end
end
Acc_dk((k-lb_k)/step_k + 1, :) = sum(acc_rnd_d, 1)/N_rnd; % N_rnd次的平均值
end
%% 画图
Color = [237,177,32;
    217,83,25;
    255,153,200;
    77,190,238;
    162,20,47;
    125,46,143;
    119,172,48;
    218,179,255;
    0,114,189;
    189,167,164;
    235,181,156;
    9,11,97;
    0,255,0;
    0,0,0]./255;
node_shape =['+-';'x-';'<-';'>-';'v-';'^-';'o-';'s-';'p-';'*-'; 'o:';'*:';'s:';'.-';];
figure;
for k=1:len_k
    plot(lb_d:ub_d, Acc_dk(k,:), node_shape(k,:), 'MarkerFaceColor','w','Linewidth', 1.5, 'Color', Color(k,:)); hold on;
end
xlabel('Dims', 'Fontsize', 16);
ylabel('Recognition rate (%)', 'Fontsize', 16);
legend('$k=10$', '$k=15$', '$k=20$', '$k=25$','$k=30$','$k=35$','$k=40$',...
    '$k=45$', '$k=50$','$k=55$', '$k=60$','$k=65$', '$k=70$','$k=75$',...
    'Interpreter','latex', 'Location', 'southeast', 'Fontsize', 16);
% legend('$k=10$','$k=20$','$k=30$','$k=40$','$k=50$','$k=60$','$k=70$',...
%     '$k=80$','$k=90$','$k=100$','$k=110$',...
%     'Interpreter','latex', 'Location', 'southeast', 'Fontsize', 16);
% legend('$k=10$','$k=20$','$k=30$','$k=40$','$k=50$','$k=60$','$k=70$',...
%     '$k=80$','$k=90$','$k=100$','$k=110$','$k=120$','$k=130$','$k=140$',...
%     'Interpreter','latex','Location','southeast','Fontsize', 16);
% legend('$k=5$','$k=20$','$k=35$','$k=50$','$k=65$','$k=80$','$k=95$',...
%     '$k=110$','$k=125$','$k=140$','$k=155$','$k=170$','$k=185$',...
%     'Interpreter','latex', 'Location', 'southeast', 'Fontsize', 16);
hold off;

