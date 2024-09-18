clear;
load("Yale_32.mat");
X = fea'; clear fea; % X:D*N  gnd:N*1

[D,N] = size(X);
N_class = 15; N_per_class = 11; scale = 32;
Faces = zeros([scale*N_per_class, scale*N_class]); 
for j=0:N-1
    face = zeros([scale, scale]);
    for k=1:scale
        face(:,k) = X((k-1)*scale+1:k*scale,j+1);
    end
    topleft_row = mod(j, N_per_class)*scale + 1;
    topleft_col = floor(j/N_per_class)*scale + 1;
    Faces(topleft_row:topleft_row-1+scale, topleft_col:topleft_col-1+scale) = face;
end
figure; imshow(Faces, []);