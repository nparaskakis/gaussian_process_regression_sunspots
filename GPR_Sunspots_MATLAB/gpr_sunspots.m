%% Gaussian Process Regression
%% An application on sunspots data-set



%% Nikolaos Paraskakis
%% I.D.: 2018030027



%% Initialization

clear;
clc;
close all;



%% Read data

[x_observed,y_observed,y_std] = read_sunspots_data('data.csv');



%% Parameters for script

kernel = 'rationalquadratic';
normalization = 0;
clever_partitioning = 1;
p = 0.1;
lossfun = @(y_true,y_pred,W) rrmse(y_true,y_pred,W); % RRMSE
%lossfun = 'mse'; % MSE
maxEvalNum = 10;
maxTimeSec = 5*60;



%% Normalize data

[y_observed_normalized,y_observed_mean,y_observed_std] = normalize_data(y_observed);

if (normalization == 0)
    y_observed_normalized = y_observed;
end



%% GPR Model (Clever partitioning)

if (clever_partitioning == 1)
    % Partition dataset in training (1) and test (0) set
    partition = ~clever_partition(y_observed_normalized,p);
end

if (clever_partitioning == 0)
    % Partition dataset in training (1) and test (0) set
    partition = ~random_partition(y_observed_normalized,p);
end

% CV GPR Model
gprMdl = fitrgp(x_observed(partition),y_observed_normalized(partition),'KernelFunction',kernel,'Optimizer','lbfgs','OptimizeHyperparameters',{'BasisFunction','KernelScale','Sigma'},'HyperparameterOptimizationOptions',struct('Optimizer','bayesopt','ShowPlots',0,'MaxObjectiveEvaluations',maxEvalNum,'MaxTime',maxTimeSec,'AcquisitionFunctionName','expected-improvement-per-second-plus'));

% Predict test data
[y_pred,~,~] = predict(gprMdl,x_observed(~partition));

% Compute loss on test data
loss_test = loss(gprMdl,x_observed(~partition),y_observed_normalized(~partition),'LossFun',lossfun);

% Compute loss on train data
loss_train = loss(gprMdl,x_observed(partition),y_observed_normalized(partition),'LossFun',lossfun);

% Predict data on a continuous x-set
x = linspace(min(x_observed),max(x_observed),10000)';
[y_pred_cont,y_pred_std,y_pred_int] = predict(gprMdl,x,'Alpha',0.05);

% Plot model
figure();
if (normalization == 1)
    subplot(2,1,1);
end
hold on;
scatter(x_observed,y_observed_normalized,'ok','filled');
scatter(x_observed(~partition),y_observed_normalized(~partition),'og','filled');
scatter(x_observed(~partition),y_pred,'or','filled');
plot(x,y_pred_cont,'r');
patch([x;flipud(x)],[y_pred_int(:,1);flipud(y_pred_int(:,2))],'k','FaceAlpha',0.1);
hold off;
if (normalization == 1)
    str0 = strcat('Model Of Gaussian Process Regression Fit (Normalized Data)');
end
if (normalization == 0)
    str0 = strcat('Model Of Gaussian Process Regression Fit');
end
str1 = strcat('Kernel Function:',{' '},kernel);
str2 = strcat('Holdout:',{' '},num2str(p));
if (clever_partitioning == 0)
    str3 = strcat('Partitioning:',{' '},'Random');
end
if (clever_partitioning == 1)
    str3 = strcat('Partitioning:',{' '},'Clever');
end
title({str0,str1{1},str2{1},str3{1}});
legend({'True Train Observations','True Test Observations','Predicted Test Observations','Continuous Predicted Line','Prediction Intervals of 95% Confidence Level'},'Location','best')

% Plot denormalized model (if normalization applied before training)
if (normalization == 1)
    subplot(2,1,2);
    hold on;
    scatter(x_observed,y_observed,'ok','filled');
    scatter(x_observed(~partition),y_observed(~partition),'og','filled');
    scatter(x_observed(~partition),y_pred*y_observed_std + y_observed_mean,'or','filled');
    plot(x,y_pred_cont*y_observed_std + y_observed_mean,'r');
    patch([x;flipud(x)],[y_pred_int(:,1)*y_observed_std + y_observed_mean;flipud(y_pred_int(:,2)*y_observed_std + y_observed_mean)],'k','FaceAlpha',0.1);
    hold off;
    str0 = strcat('Model Of Gaussian Process Regression Fit (Denormalized Data)');
    str1 = strcat('Kernel Function:',{' '},kernel);
    str2 = strcat('Holdout:',{' '},num2str(p));
    if (clever_partitioning == 0)
        str3 = strcat('Partitioning:',{' '},'Random');
    end
    if (clever_partitioning == 1)
        str3 = strcat('Partitioning:',{' '},'Clever');
    end
    title({str0,str1{1},str2{1},str3{1}});
    legend({'True Train Observations','True Test Observations','Predicted Test Observations','Continuous Predicted Line','Prediction Intervals of 95% Confidence Level'},'Location','best')
end