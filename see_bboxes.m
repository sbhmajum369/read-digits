''' Matlab code for single digit extraction along with their labels'''

clear;
clc;
load digitStruct.mat
fid=fopen('Labels.txt','w');
k=0;
folder='D:\Images'; % Storing folder for extracted single digits
n=input('Number of images to consider:')
for i = 1:n
    im = imread([digitStruct(i).name]);
    for j = 1:length(digitStruct(i).bbox)
        [height, width] = size(im);
        aa = max(digitStruct(i).bbox(j).top+1,1);
        bb = min(digitStruct(i).bbox(j).top+digitStruct(i).bbox(j).height, height);
        cc = max(digitStruct(i).bbox(j).left+1,1);
        dd = min(digitStruct(i).bbox(j).left+digitStruct(i).bbox(j).width, width);
        image=im(aa:bb, cc:dd, :);
        [Y,X,D]=size(image);
        % X is the width of the image
        if X>=7
            k=k+1;
            imshow(image);
            baseFileName = sprintf('%d.png', k);
            fullFileName = fullfile(folder, baseFileName);
            imwrite(image, fullFileName);
            fprintf(fid,'%d',digitStruct(i).bbox(j).label );
            fprintf(fid,'\n');
        end
        %pause;
        
    end
    
end
fclose(fid);
