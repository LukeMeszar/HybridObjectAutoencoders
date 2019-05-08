function idx = cluster(X, k)
    if ~isa(X, 'double')
        X = double(X);
    end
    m = size(X, 1);
    n = size(X, 2);
    
    idx = zeros(m, 1);

    iter = 0;
    
    while true        
        old_idx = idx;
        
        for i=1:m
            diff = X(i,:) - centers;
            distance = sqrt(diff(:,1).^2 + diff(:,2).^2);
            [min_val, min_ind] = min(distance);
            idx(i) = min_ind;
        end
        
        if idx == old_idx
            break;
        end

        centers_sum = zeros(k, n);
        centers_members = zeros(k,1);
        
        for i=1:m
            centers_sum(idx(i),:) = centers_sum(idx(i),:) + X(i,:);
            centers_members(idx(i)) = centers_members(idx(i)) + 1;
        end
        for i=1:k
            centers(i,1) = centers_sum(i,1) / centers_members(i);
            centers(i,2) = centers_sum(i,2) / centers_members(i);
        end
        
        iter = iter + 1;
        if iter > 100
            break;
        end
    end
end