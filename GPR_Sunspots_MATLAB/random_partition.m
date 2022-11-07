function partition = random_partition(y,p)
    
    n = length(y);
    
    cvp = cvpartition(n,'Holdout',p);
    
    partition = cvp.test;
    
end