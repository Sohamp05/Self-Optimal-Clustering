% To implement Self Optimal Clustering(SOC) technique 
% for images containing RGB data points or for any (n x k) matrix
% WITH ALL THE DATA POINTS INCLUDED

clc
clear
disp('input the image name in the format --> imread(image name, image format); ');
a = input('Enter image data : ');
[f, h, k] = size(a);
nk = input('no. of clusters required : ');                      % nk is no. of clusters required

for j = 1:f
    for i = 1:h
        R(j,i) = a(j,i,1);                                  % extracting R value from original matrix a
        G(j,i) = a(j,i,2);                                  % extracting G value from original matrix a
        B(j,i) = a(j,i,3);                                  % extracting B value from original matrix a
    end
end
n = f*h;                                                    % n is total no. of data points
for j = 1:n
    x(j,:) = [R(j),G(j),B(j)];
end                                                         % So x is a (n x 3) matrix for RGB 

[fac]=factorcal(x,nk,1);
[result] = soc(x,nk,fac);
x = double(x);
delta1 = 0.1;             % you’ll tune this per-dataset
delta2 = delta1 / 2;      % a common heuristic
[res1] = imc2(x, nk, 1);
[res2] = imc2(x, nk, (nk/(nk+1)));


[s] = silhouette(double(x), result.idx);
[S, GSS] = slht(s, result.idx, result.n, result.m, nk);
GSS

s1 = silhouette(double(x), res1.idx);
[S1, GSI1] = slht(s1, res1.idx, res1.n, res1.m, nk);
GSI1
s2 = silhouette(double(x), res2.idx);
[S2, GSI2] = slht(s2, res2.idx, res2.n, res2.m, nk);
GSI2
%[PI, SI] = valid(result.dd,result.cc_norm,result.part.^2,nk)
% [ADI] = adu(result.clst,nk,result.m)
metrics = [GSS, GSI1, GSI2];
methods = {'SOC','IMC1','IMC2'};
results = {result, res1, res2};

figure('units','normalized','position',[0.1 0.1 0.8 0.6]);

for m = 1:3
    res    = results{m};
    labels = res.idx;
    
    % compute cluster means
    cmap = zeros(nk,3);
    for v = 1:nk
        pts = x(labels==v,:);
        if ~isempty(pts)
            cmap(v,:) = mean(pts,1);
        end
    end
    
    % rebuild segmented image
    seg_rgb = cmap(labels,:);
    seg_img = reshape(seg_rgb, [f, h, 3]);
    seg_img = uint8(seg_img);
    
    % original
    subplot(2,3,m);
    imshow(a);
    title([methods{m} ' — Original']);
    
    % segmented + overlay
    subplot(2,3,m+3);
    imshow(seg_img);
    title([methods{m} ' — Segmented']);
    hold on;
    % put text in top-left corner (10,10 pixels from top-left)
    text(10, 10, sprintf('GSI = %.3f', metrics(m)), ...
         'Color','yellow','FontSize',14,'FontWeight','bold', ...
         'HorizontalAlignment','left','VerticalAlignment','top');
    hold off;
end