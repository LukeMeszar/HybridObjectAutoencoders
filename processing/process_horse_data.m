function [] = process_horse_data(arg1,arg2)
disp("First argument must be path to dataset with an annotation\list.txt file, second argument must be size of output images");
mkdir("horse");
S = dir(fullfile(strcat(arg1,'\images\'),'*.JPEG'));
for i = 1:numel(S)
    path = fullfile(strcat(arg1,'\images\'),S(i).name);
    S(i).name = S(i).name(1:end-5);
    I = imread(path);
    S(i).data = I;
    
    try
        filename_seg = strcat(S(i).name,".gt.png");
        seg = imread(strcat(strcat(arg1,'\annotations\'), filename_seg));
    catch
        filename_seg = strcat(S(i).name,".bb.png");
        seg = imread(strcat(strcat(arg1,'\annotations\'), filename_seg));
    end
    
    segmented_img = I .* seg;
    img_name = strcat(S(i).name, ".png");
    segmented_img = imresize(segmented_img, [arg2, arg2]);
    imwrite(segmented_img, strcat("horse\", img_name));
end


