clc
clear
close all

%%
% Load feature matrix.
load('mixM.mat');
Label = Labels';

% Apply chi-square tests to feature selection.
active_CHI = 1;
if active_CHI
    [idx, scores] = fscchi2(Features,Label);
    Features = Features(:,idx(1:110));
    % Save chi-square tests order.
    save('idx.mat','idx')
end

% Train SVM model on training data.
rng(1); % For reproducibility

t = templateSVM('KernelScale','auto','KernelFunction','polynomial',...
    'PolynomialOrder',3);
SVMModel = fitcecoc(Features,Labels,'Learners',t);

% Save trained model.
save('TrainedModel.mat','SVMModel')
