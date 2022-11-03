function [y_normalized,mu,sigma] = normalize_data(y)

    mu = mean(y);
    sigma = std(y);

    y_normalized = normalize(y);

end