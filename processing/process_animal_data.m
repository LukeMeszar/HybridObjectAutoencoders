function [] = process_animal_data(arg1,arg2)
    disp("First argument must be path to dataset with an annotation\list.txt file, second argument must be size of output images");
    fileID = fopen(strcat(arg1,'\annotations\list.txt'),'r');
    A = textscan(fileID, '%s', 'Delimiter','\n');
    fclose(fileID);
    data_size = size(A{1,1});

    mkdir("dog");
    mkdir("cat");

    for i=1:data_size(1)
        words = strsplit(A{1,1}{i,1});
        filename_raw = words{1,1};
    
        filename_img = strcat(filename,".jpg");
    
        filename = strip(filename_raw,'right','0');
        filename_seg = strcat(filename_raw,".png");
    
        filename_img_raw = strcat(filename_raw,".jpg");

        if isstrprop(filename(1),'upper') == 1
            try
                img = imread(strcat(strcat(arg1,'\annotations\trimaps\'), filename_seg));
            catch
                continue
            end
            img = double(img);
            img = abs(img-2);

            try
                img2 = imread(strcat(strcat(arg1,'\images\images\'), filename_img_raw));
            catch
                img2 = imread(strcat(strcat(arg1,'\images\images\'), filename_img));
            end
 
            img2 = double(img2);
            img2 = img2 .* img;
            img_segmented = uint8(img2);
            img_segmented = imresize(img_segmented, [arg2, arg2]);
            imwrite(img_segmented, strcat("cat\", filename_seg));
        else
            try
                img = imread(strcat(strcat(arg1,'\annotations\trimaps\'), filename_seg));
            catch
                continue
            end
            img = double(img);
            img = abs(img-2);

            try
                img2 = imread(strcat(strcat(arg1,'\images\images\'), filename_img_raw));
            catch
                img2 = imread(strcat(strcat(arg1,'\images\images\'), filename_img));
            end
     
            img2 = double(img2);
            img2 = img2 .* img;
            img_segmented = uint8(img2);
            img_segmented = imresize(img_segmented, [arg2, arg2]);
            imwrite(img_segmented, strcat("dog\", filename_seg));
        end
    end
end
