clear;
clc;
clf;
close all;


% Data points
odeRef = load("DCmotor_ODE45_reference");
dataPoints = 3;
dataID = round(linspace(1, length(odeRef.t), dataPoints));
tData = dlarray(odeRef.t(dataID)', "CB");
yData = dlarray(odeRef.y(dataID, :)', "CB");

% Collocation points
numCollocation = 50;
tMax = odeRef.t(end);
t = dlarray(linspace(0, tMax, numCollocation), "CB");

% Initial parameters
J_init = 0.01;
b_init = 0.1;

% True parameters
J_true = odeRef.J;
b_true = odeRef.b;

% Neural Network
layers = [
    featureInputLayer(1)
    fullyConnectedLayer(16)
    tanhLayer
    fullyConnectedLayer(16)
    tanhLayer
    fullyConnectedLayer(16)
    tanhLayer
    fullyConnectedLayer(3)
];
net = dlnetwork(layers);

% Initial conditions
y0 = dlarray(odeRef.y(1, :)', "CB");
t0 = dlarray(odeRef.t(1), "CB");

% lambda values
lambdaPhysics = 1;
lambdaData = 1;

% Adam optimizer parameters
numAdamIterations = 10000;
learningRate = 1e-3;
averageGrad = [];
averageSqGrad = [];
lossHistoryAdam = zeros(numAdamIterations, 1);

averageGradJ = [];
averageSqGradJ = [];

averageGradb = [];
averageSqGradb = [];

% Estimated parameters
J_est = dlarray(J_init);
b_est = dlarray(b_init);

% Training
trainingTimer = tic;


for iter = 1:numAdamIterations
    [loss, gradients, physicsLoss, initialConditionLoss, dataLoss, gradJ, gradb] = dlfeval(...
        @DCmotor_modelLoss_learnParam, net, t, t0, tData, yData,...
        odeRef.R, odeRef.L, J_est, b_est,...
        odeRef.Kt, odeRef.Ke, odeRef.U0,...
        odeRef.load_torque, y0,...
        lambdaPhysics, lambdaData);
    
    [net, averageGrad, averageSqGrad] = adamupdate(net, gradients, averageGrad, averageSqGrad, iter, learningRate);
    [J_est, averageGradJ, averageSqGradJ] = adamupdate(J_est, gradJ, averageGradJ, averageSqGradJ, iter, learningRate);
    J_estFinal = extractdata(J_est);
    [b_est, averageGradb, averageSqGradb] = adamupdate(b_est, gradb, averageGradb, averageSqGradb, iter, learningRate);
    b_estFinal = extractdata(b_est);
    
    lossHistoryAdam(iter) = extractdata(loss);
    
    if mod(iter, 100) == 0 || iter == 1
        disp("Iteration: " + iter + ...
            ", Total Loss: " + extractdata(loss) + ...
            ", Physics Loss: " + extractdata(physicsLoss) + ...
            ", Initial Condition Loss: " + extractdata(initialConditionLoss) + ...
            ", Data Loss: " + extractdata(dataLoss));
    end
end



trainingTimeSeconds = toc(trainingTimer);
fprintf("Training time: %.2f seconds\n", trainingTimeSeconds);

figure

semilogy(1:length(lossHistoryAdam), lossHistoryAdam, "LineWidth",2)
hold on


yPINN = extractdata(predict(net, dlarray(odeRef.t', "CB")))';

rmse = sqrt(mean((yPINN - odeRef.y).^2, 1));

fprintf('RMSE áram:         %.6g A\n', rmse(1));
fprintf('RMSE szögsebesség: %.6g rad/s\n', rmse(2));
fprintf('RMSE pozíció:      %.6g rad\n', rmse(3));


fprintf('J becslés: %.6g kg*m^2, igaz érték: %.6g kg*m^2\n', ...
    J_estFinal, double(J_true));

fprintf('b becslés: %.6g Nm*s, igaz érték: %.6g Nm*s\n', ...
    b_estFinal, double(b_true));

figure;

subplot(3,1,1)
plot(odeRef.t, odeRef.y(:,1), 'b-', odeRef.t, yPINN(:,1), 'r--')
ylabel('Áram [A]')
legend('ODE45', 'PINN')
grid on

subplot(3,1,2)
plot(odeRef.t, odeRef.y(:,2), 'b-', odeRef.t, yPINN(:,2), 'r--')
ylabel('Szögsebesség [rad/s]')
legend('ODE45', 'PINN')
grid on

subplot(3,1,3)
plot(odeRef.t, odeRef.y(:,3), 'b-', odeRef.t, yPINN(:,3), 'r--')
ylabel('Pozíció [rad]')
xlabel('Idő [s]')
legend('ODE45', 'PINN')
grid on

disp("Lefutott minden :)");