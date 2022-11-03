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


%% Normalize data
[y_observed_normalized,y_observed_mean,y_observed_std] = normalize_data(y_observed);

%y_observed_normalized = y_observed;
%% CVGPR Model 1
lossfun = @(y_true,y_pred,W) rrmse(y_true,y_pred,W);

gprMdl1 = fitrgp(x_observed,y_observed_normalized,'KernelFunction','squaredexponential','Optimizer','lbfgs','OptimizeHyperparameters',{'BasisFunction','KernelScale','Sigma'},'HyperparameterOptimizationOptions',struct('Optimizer','bayesopt','ShowPlots',0,'MaxObjectiveEvaluations',50,'MaxTime',60,'AcquisitionFunctionName','expected-improvement-per-second-plus'));

cvgprMdl1 = crossval(gprMdl1,'Holdout',0.1);

y_pred_1_fold = kfoldPredict(cvgprMdl1);

loss_1 = kfoldLoss(cvgprMdl1,'LossFun',lossfun);

x = linspace(min(x_observed),max(x_observed),10000)';

[y_pred_1,y_pred_std_1,y_pred_int_1] = predict(cvgprMdl1.Trained{1},x,'Alpha',0.05);

loss_11 = loss(cvgprMdl1.Trained{1},x_observed,y_observed_normalized,'LossFun',lossfun);

Data = y_observed_normalized;[Maxima,MaxIdx] = findpeaks(Data);
DataInv = 1.01*max(Data) - Data;
[Minima,MinIdx] = findpeaks(DataInv);

figure();
hold on;
scatter(x_observed,y_observed_normalized,'ok','filled');
scatter(x_observed(cvgprMdl1.Partition.test),y_observed_normalized(cvgprMdl1.Partition.test),'og','filled');
scatter(x_observed(cvgprMdl1.Partition.test),y_pred_1_fold(cvgprMdl1.Partition.test),'or','filled');

scatter(x_observed(MinIdx),y_observed_normalized(MinIdx),'g','pentagram','LineWidth',5);

scatter(x_observed(MaxIdx),y_observed_normalized(MaxIdx),'r','pentagram','LineWidth',5);

plot(x,y_pred_1,'r');
patch([x;flipud(x)],[y_pred_int_1(:,1);flipud(y_pred_int_1(:,2))],'k','FaceAlpha',0.1);
hold off;
title('GPR Fit');



figure();
hold on;
scatter(x_observed,y_observed,'ok','filled');
scatter(x_observed(cvgprMdl1.Partition.test),y_observed(cvgprMdl1.Partition.test),'og','filled');
scatter(x_observed(cvgprMdl1.Partition.test),y_pred_1_fold(cvgprMdl1.Partition.test)*y_observed_std + y_observed_mean,'or','filled');
plot(x,y_pred_1*y_observed_std + y_observed_mean,'r');
patch([x;flipud(x)],[y_pred_int_1(:,1)*y_observed_std + y_observed_mean;flipud(y_pred_int_1(:,2)*y_observed_std + y_observed_mean)],'k','FaceAlpha',0.1);
hold off;
title('GPR Fit');

%% CVGPR Model 2

cvgprMdl2 = fitrgp(x_observed,y_observed_normalized,'Optimizer','lbfgs','OptimizeHyperparameters','auto','HyperparameterOptimizationOptions',struct('Optimizer','randomsearch','MaxObjectiveEvaluations',100,'ShowPlots',0));

[y_pred_2,y_pred_std_2,y_pred_int_2] = kfoldPredict(cvgprMdl2,'Alpha',0.05);

x = linspace(min(x_observed),max(x_observed),10000)';

[y_pred_1,y_pred_std_1,y_pred_int_1] = predict(gprMdl1,x,'Alpha',0.05);

figure();
hold on;
scatter(x_observed,y_observed_normalized,'ob','filled');
plot(x,y_pred_1,'r');
patch([x;flipud(x)],[y_pred_int_1(:,1);flipud(y_pred_int_1(:,2))],'k','FaceAlpha',0.1);
hold off;
title('GPR Fit');


%% CVGPR Model 3
cvgprMdl2 = fitrgp(x_observed,y_observed_norm,'OptimizeHyperparameters','all');

y_pred = kfoldPredict(cvgprMdl2);

loss = kfoldLoss(cvgprMdl1);

aa = cvgprMdl2.Trained(1);
aa = aa{1};
bb = aa.Partition.test;

[ypred1,~,yint1] = predict(aa,x,'Alpha',0.05);

figure();
hold on
scatter(x_observed,y_observed_norm,'xb') % Observed data points
scatter(x_observed(cvgprMdl1.Partition.test),y_observed_norm(cvgprMdl1.Partition.test),'ok','filled')
scatter(x_observed(cvgprMdl1.Partition.test),y_pred(cvgprMdl1.Partition.test),'or','filled')                  % GPR predictions
patch([x;flipud(x)],[yint1(:,1);flipud(yint1(:,2))],'k','FaceAlpha',0.1); % Prediction intervals
hold off

title('GPR Fit of Noise-Free Observations')


figure();
hold on
scatter(x_observed,y_observed2,'xr') % Observed data points
plot(x,ypred2,'g')                   % GPR predictions
patch([x;flipud(x)],[yint2(:,1);flipud(yint2(:,2))],'k','FaceAlpha',0.1); % Prediction intervals
hold off
title('GPR Fit of Noisy Observations')
