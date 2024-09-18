clear;
% warning off;
%% 载入 MNIST 数据集
load('Yale_64.mat'); % fea: 4000*784, gnd: 4000*1, trainIdx,testIdx: 4000*1
X = fea'; clear fea; % X:D*N  N=165
X = X./255;
classes = unique(gnd); % [0; 1; ...; 9]
n_class = length(classes); % 类别数目
colors = ["r","g","b","c","m","y","k","#D95319","#7E2F8E","#0072BD"];
%% 
[D,N] = size(X);
% off = 64;
% for i=1:5
%     face = zeros([off,off]);
%     for j=1:off
%         face(:,j)=X((j-1)*off+1:off*j,i);
%     end
% %     Mx=max(face(:));
% %     mn=min(face(:));
% %     face=(face-mn)./(Mx-mn);
%     figure;imshow(face);
% end
t = 3e5; % 
k = 5;

[W] = LapFace_my(X, k, t);

off=64;
for i=1:10
    face = zeros([off,off]);
    for j=1:off
        face(:,j)=W((j-1)*off+1:off*j,i);
    end
    Mx=max(face(:));
    mn=min(face(:));
    face=(face-mn)./(Mx-mn);
    figure;imshow(face);
    imwrite(face, ['./LapFace', num2str(i),'_k', num2str(k), '_t', num2str(t), '.png']);
end
