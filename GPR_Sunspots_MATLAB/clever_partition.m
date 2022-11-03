function partition = clever_partition(y,p)
    %hajaaaaaaa
    n = length(y);
    
    cvp = cvpartition(n,'Holdout',p);
    
    test_size = length(find(cvp.test == 1));
    
    [~,MaxIdx] = findpeaks(y);
    yInv = 1.01*max(y) - y;
    [~,MinIdx] = findpeaks(yInv);
    
    partition = ones(n,1);
    
    partition(MinIdx) = 0;
    
    partition(MaxIdx) = 0;
    
    one_indicies = find(partition == 1);
    
    rem_zero_indicies = randperm(length(one_indicies),length(find(partition == 1)) - test_size);
    
    partition(one_indicies(rem_zero_indicies)) = 0;
    
end