function [y_original] = denormalize_data(y_normalized,mu,sigma)
    
    y_original = sigma*y_normalized + mu;

end
