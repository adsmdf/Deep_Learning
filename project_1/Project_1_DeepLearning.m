%Adam Solimani
%Deep Learning
%Mini Project 1

%% Housekeeping
clc; close all; clear;

%% Load Database
load bodyfat_dataset
[x,t] = bodyfat_dataset;

x = bodyfatInputs;
t = bodyfatTargets;

%% Correlation Coefficient of each input with the output
num_matrices = 13;
for i = 1:num_matrices
    
    % Calculate the correlation matrix for each i
    correlation_matrix = corrcoef(x(i,:), t);
    Part_A{i} = correlation_matrix(1,2);
end

%% Training Algorithm
%trainFcn = 'trainlm';  % Levenberg-Marquardt backpropagation.
trainFcn = 'trainbr'; % Bayesian Regularization backpropagation
%trainFcn = 'trainscg'; % Scaled conjugate gradient backpropagation

%% Main
%running 10 times
num_runs = 10;
for i = 1:num_runs
%Create Network
    Neurons = 15; %Number of Neurons
    net = fitnet(Neurons,trainFcn);
    %net = fitnet([8 5], trainFcn) %Each number represents the number of nodes per the number of layers you want
    %net.performParam.regularization = ; %Change weight decay
%Training the Network
    [net,tr] = train(net,x,t);
%Splitting the Dataset 70% Training, 15% Validation, 15% Test
    net.divideParam.trainRatio = 70/100;
    net.divideParam.valRatio = 15/100;
    net.divideParam.testRatio = 15/100;
%Saving Tr data
    tr_save{i} = tr;
%Part B.1 saving best values
    mse{i} = [tr.best_perf tr.best_tperf tr.best_vperf]; 
end

%% Part B.1 Finding mean and var
best_perf = [];
best_tperf = [];
best_vperf = [];
for i = 1:10
    best_perf(i) = mse{i}(1,1);
    best_tperf(i) = mse{i}(1,2);
    best_vperf(i)= mse{i}(1,3);
end
% Means
mean_best_perf = mean(best_perf);
mean_best_tperf = mean(best_tperf);
mean_best_vperf = mean(best_vperf);
% Variance
var_best_perf = var(best_perf);
var_best_tperf = var(best_tperf);
var_best_vperf = var(best_vperf);
%% Testing the Network
y = net(x);
e = gsubtract(t,y);
performance = perform(net,t,y)

% tInd = tr.testInd;
% tstOutputs = net(x(:,tInd));
% tstPerform = perform(net,t(tInd),tstOutputs)

%% View the Network
view(net)