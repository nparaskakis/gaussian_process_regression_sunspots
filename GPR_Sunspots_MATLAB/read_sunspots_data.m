function [x,y,y_std] = read_sunspots_data(csv_filename)

    data = readmatrix(csv_filename);

    data(:,4:5) = [] ;
    
    x = data(:,1);
    y = data(:,2);
    y_std = data(:,3);
    
    true_std = y_std(y_std ~= -1);
    
    mu = mean(true_std);
    
    y_std(y_std == -1) = mu;
    
end