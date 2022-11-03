function loss_val = rrmse(y_true,y_pred,W) %#ok<INUSD>
    
    y_true = y_true(~isnan(y_pred));
    y_pred = y_pred(~isnan(y_pred));
    
    loss_val = sqrt(immse(y_true,y_pred))/sqrt(sum(y_pred.^2));
    
end