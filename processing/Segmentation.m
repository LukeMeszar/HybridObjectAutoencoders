function segments = Segmentation(img, k)
    height = size(img, 1);
    width = size(img, 2);
    
    features = zeros(height, width, 5);
    features(:,:,1:3) = double(img);
    features(:,:,4) = repmat((1:width),height,1)*5;
    features(:,:,5) = repmat((1:height)',1,width)*5;
    
    points = reshape(features, [], size(features, 3));
    idx = cluster(points, k);
    idx = reshape(idx, size(features, 1), size(features, 2));
    idx = imresize(idx, [height, width], 'nearest');
    segments = MakeSegments(img, idx);
end