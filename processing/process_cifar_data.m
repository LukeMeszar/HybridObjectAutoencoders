function [] = process_cifar_data(arg1)
    %CIFAR
    disp("Argument must be path to .mat data");
    load(arg1);
    data_size = size(data);
    im_all = {};
    im=zeros(32,32,3);
    im_index = 1;
    for cpt=1:data_size(1) 
        R=data(cpt,1:1024);
        G=data(cpt,1025:2048);
        B=data(cpt,2049:3072);
        k=1;
        for x=1:32
            for i=1:32
                im(x,i,1)=R(k);
                im(x,i,2)=G(k);
                im(x,i,3)=B(k);
                k=k+1;
            end
        end  
        im=uint8(im);
        img = im;
        k = 2;
        segments = Segmentation(img, k);             
        num_clusters = size(segments);
        for i = 1:num_clusters(2)
            if segments(i).img(16,16,1) < 240
            cluster = segments(i).img;
            end
        end
        im_all{im_index} = cluster;
        im_index = im_index +1;
    end
        
    mkdir("airplane");
    mkdir("automobile");
    mkdir("bird");
    mkdir("cat");
    mkdir("deer");
    mkdir("dog");
    mkdir("frog");
    mkdir("horse");
    mkdir("ship");
    mkdir("truck");
        
    for i=1:data_size(1)
        if labels(i)==0
            dir_name = "airplane";
        elseif labels(i)==1
            dir_name = "automobile";
        elseif labels(i)==2
            dir_name = "bird";
        elseif labels(i)==3
            dir_name = "cat";
        elseif labels(i)==4
            dir_name = "deer";
        elseif labels(i)==5
            dir_name = "dog";
        elseif labels(i)==6
            dir_name = "frog";
        elseif labels(i)==7
            dir_name = "horse";
        elseif labels(i)==8
            dir_name = "ship";
        elseif labels(i)==9
            dir_name = "truck";
        end
        
        img = im_all{i};
        img_name = strcat("img_",num2str(i));
        img_name = strcat(img_name, ".png");
        img_file_name = fullfile(dir_name, img_name);
        imwrite(img, img_file_name);
    end

end