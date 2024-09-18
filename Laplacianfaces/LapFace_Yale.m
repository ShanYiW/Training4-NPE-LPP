clear;
load('Yale_32.mat');
Yale = fea'; clear fea; % X:D*N  gnd:N*1
[D,N] = size(Yale);
%%
for t=[3e5]
L = 6; % 固定
N_train = L*15; % 训练样本数
N_test = N-N_train; % 测试样本数
trainset = zeros([D, N_train]); % 训练集
testset = zeros([D,N-N_train]); % 测试集
gnd_train = zeros([N_train,1]); 
gnd_test = zeros([N_test,1]); % gnd_test(i)=第i个测试样本的类别
N_class = 15; % 类别数
N_per_class = 11; % 每类样本数

lb_d=10; ub_d=65;
len_d = ub_d-lb_d+1;
lb_k=5; ub_k=70; step_k=5; % ub_k<=90
len_k = (ub_k-lb_k)/step_k + 1;

Acc_dk = zeros([len_k, len_d]);
N_rnd = 20; % 重复次数
for k=lb_k:step_k:ub_k
    acc_rnd_d = zeros([N_rnd, len_d]); % 
    for rnd = 1:N_rnd
        %% 生成trainset, train_gnd, testset, test_gnd
        i_tr = 0; i_te = 0;
        for j=1:N_class % 15个人
            train_idx = sort(randperm(N_per_class,L)); % 升序排序
            trainset(:,i_tr+1:i_tr+L) = Yale(:, [(j-1)*N_per_class + train_idx]); % 
            gnd_train(i_tr+1:i_tr+L) = gnd([(j-1)*N_per_class + train_idx]);
            i_tr = i_tr + L;
            test_idx = setdiff(1:N_per_class,train_idx); % 集合作差: 全集-train_idx
            testset(:,i_te+1:i_te+N_per_class-L) = Yale(:,[(j-1)*N_per_class + test_idx]);
            gnd_test(i_te+1:i_te+N_per_class-L) = gnd([(j-1)*N_per_class + test_idx]);
            i_te = i_te + N_per_class - L;
        end
        %% 运行 LaplacianFaces
        trainset = trainset - repmat(mean(trainset,2), [1,N_train]);
        testset = testset - repmat(mean(testset,2), [1,N_test]);
        [Wpca,Wlpp] = LapFace_my(trainset, k, t); % E: D*D
%         opt=[]; opt.NeighborMode = 'Supervised'; opt.WeightMode = 'HeatKernel'; 
%         opt.bSelfConnected = 1; opt.gnd=gnd_train; opt.t = 3e3; 
%         W = constructW(trainset',opt);
%         W = full(W); for j=1:N_train; W(j,j)=0; end
%         opt.k = k; opt.PCARatio=1; opt.ReducedDim=ub_d;
%         [E, Eval] = LPP(W,opt, trainset');
        %% 计算分类准确度
        for d=lb_d:ub_d
            Y_train = Wlpp(:,1:d)'*Wpca'*trainset;
            Y_test  = Wlpp(:,1:d)'*Wpca'*testset;
            acc = 0;
            for j=1:N_test
                y = Y_test(:,j); % d*1
%                 pred = classifier_knn(y, Y_train, gnd_train, 7, N_class);
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
ylim([0.2,0.58]);
xlabel('Dims', 'Fontsize', 16);
ylabel('Recognition rate (%)', 'Fontsize', 16);
legend('$k=5$', '$k=10$', '$k=15$', '$k=20$', '$k=25$','$k=30$','$k=35$',...
    '$k=40$','$k=45$', '$k=50$','$k=55$', '$k=60$','$k=65$', '$k=70$',...
    'Interpreter','latex', 'Location', 'southeast', 'Fontsize', 16);
hold off;
end