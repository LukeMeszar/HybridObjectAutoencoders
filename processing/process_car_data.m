function [] = process_car_data(arg1, arg2)
    disp("First argument must be path to .mat data, second argument must be size of output images");
    load(arg1);
    mkdir("cars");
    data_size = size(data);
    for i=1:data_size(1)
        img_name = annotations(i).fname;
        img = imread(strcat("C:\Users\max_schwarz\Documents\hw\deeplearning\final_project\cars\cars_train\",img_name));
        img(:,1:annotations(i).bbox_x1,:) = 0;
        img(:,annotations(i).bbox_x2:end,:) = 0;
    
        img(1:annotations(i).bbox_y1,:,:) = 0;
        img(annotations(i).bbox_y2:end,:,:) = 0;
    
        img = imresize(img, [arg2,arg2]);
        imwrite(img, strcat("cars/",strcat(img_name(1:end-4),".png")));
    end
end