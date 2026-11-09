clear;
clc;
clf;
close all;


% Data points
odeRef = load("DCmotor_ODE45_reference_DynamicLoad.mat");
dataPoints = 200;
dataID = round(linspace(1, length(odeRef.t), dataPoints));
tData = dlarray(odeRef.t(dataID)', "CB");
yData = dlarray(odeRef.y(dataID, 1:2)', "CB");

% Collocation points
numCollocation = 500;
tMax = odeRef.t(end);
t = dlarray(linspace(0, tMax, numCollocation), "CB");


% Neural Network
layers = [
    featureInputLayer(1)
    fullyConnectedLayer(16)
    tanhLayer
    fullyConnectedLayer(16)
    tanhLayer
    fullyConnectedLayer(16)
    tanhLayer
    fullyConnectedLayer(4)
];
net = dlnetwork(layers);

% Initial conditions
y0 = dlarray(odeRef.y(1, :)', "CB");
t0 = dlarray(odeRef.t(1), "CB");

% lambda values
lambdaPhysics = 1;
lambdaData = 1;
lambdaLoadChange = 2e-4;

% Adam optimizer parameters
numAdamIterations = 10000;
learningRate = 1e-3;
averageGrad = [];
averageSqGrad = [];
lossHistoryAdam = zeros(numAdamIterations, 1);


% Training
trainingTimer = tic;

for iter = 1:numAdamIterations
    [loss, gradients, physicsLoss, initialConditionLoss, dataLoss, loadChangeLoss] = dlfeval(...
        @DCmotor_modelLoss_DynamicLoad, net, t, t0, tData, yData,...
        odeRef.R, odeRef.L, odeRef.J, odeRef.b,...
        odeRef.Kt, odeRef.Ke, odeRef.U0,...
        y0,...
        lambdaPhysics, lambdaData, lambdaLoadChange);
    
    [net, averageGrad, averageSqGrad] = adamupdate(net, gradients, averageGrad, averageSqGrad, iter, learningRate);
    lossHistoryAdam(iter) = extractdata(loss);

    if mod(iter, 100) == 0 || iter == 1
        disp("Iteration: " + iter + ...
            ", Total Loss: " + extractdata(loss) + ...
            ", Physics Loss: " + extractdata(physicsLoss) + ...
            ", Initial Condition Loss: " + extractdata(initialConditionLoss) + ...
            ", Data Loss: " + extractdata(dataLoss) + ...
            ", Load Change Loss: " + extractdata(loadChangeLoss));
    end
end


trainingTimeSeconds = toc(trainingTimer);
fprintf("Training time: %.2f seconds\n", trainingTimeSeconds);

figure

semilogy(1:length(lossHistoryAdam), lossHistoryAdam, "LineWidth",2)
hold on


yPINN = extractdata(predict(net, dlarray(odeRef.t', "CB")))';

rmse = sqrt(mean((yPINN(:, 1:3) - odeRef.y).^2, 1));
rmseLoadTorq = sqrt(mean((yPINN(:, 4) - odeRef.load_torque).^2, 1));

fprintf('RMSE áram:         %.6g A\n', rmse(1));
fprintf('RMSE szögsebesség: %.6g rad/s\n', rmse(2));
fprintf('RMSE pozíció:      %.6g rad\n', rmse(3));
fprintf('RMSE terhelés:     %.6g Nm\n', rmseLoadTorq);

figure;

subplot(4,1,1)
plot(odeRef.t, odeRef.y(:,1), 'b-', odeRef.t, yPINN(:,1), 'r--')
ylabel('Áram [A]')
legend('ODE45', 'PINN')
grid on

subplot(4,1,2)
plot(odeRef.t, odeRef.y(:,2), 'b-', odeRef.t, yPINN(:,2), 'r--')
ylabel('Szögsebesség [rad/s]')
legend('ODE45', 'PINN')
grid on

subplot(4,1,3)
plot(odeRef.t, odeRef.y(:,3), 'b-', odeRef.t, yPINN(:,3), 'r--')
ylabel('Pozíció [rad]')
xlabel('Idő [s]')
legend('ODE45', 'PINN')
grid on

subplot(4,1,4)
plot(odeRef.t, odeRef.load_torque, 'b-', odeRef.t, yPINN(:,4), 'r--')
xlabel('Idő [s]')
ylabel('Terhelés [Nm]')
legend('ODE45', 'PINN')
grid on

disp("Lefutott minden :)");