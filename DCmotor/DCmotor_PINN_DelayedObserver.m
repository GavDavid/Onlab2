clear;
clc;
clf;
close all;


% Data points
odeRef = load("DCmotor_ODE45_reference_DynamicLoad.mat");
dt = odeRef.t(2) - odeRef.t(1);

iMeas = odeRef.y(:, 1)';
omegaMeas = odeRef.y(:, 2)';
U = odeRef.U0 * ones(size(iMeas));

% Bemenet: aktuális (k) és két késleltetett (k-1, k-2) érték
xInput = [
    U(3:end);         U(2:end-1);         U(1:end-2);
    iMeas(3:end);     iMeas(2:end-1);     iMeas(1:end-2);
    omegaMeas(3:end); omegaMeas(2:end-1); omegaMeas(1:end-2)
];

tValid = odeRef.t(3:end);
yValid = odeRef.y(3:end, 1:2);
loadValid = odeRef.load_torque(3:end);

xInput = dlarray(xInput, "CB");
yData = dlarray(yValid', "CB");


% Neural Network
layers = [
    featureInputLayer(9)
    fullyConnectedLayer(16)
    tanhLayer
    fullyConnectedLayer(16)
    tanhLayer
    fullyConnectedLayer(16)
    tanhLayer
    fullyConnectedLayer(3)
];
net = dlnetwork(layers);

% lambda values
lambdaPhysics = 1;
lambdaData = 1;

% Adam optimizer parameters
numAdamIterations = 10000;
learningRate = 1e-3;
averageGrad = [];
averageSqGrad = [];
lossHistoryAdam = zeros(numAdamIterations, 1);


% Training
trainingTimer = tic;

for iter = 1:numAdamIterations
    [loss, gradients, physicsLoss, dataLoss] = dlfeval(...
        @DCmotor_modelLoss_DelayedObserver, net, xInput, yData, dt,...
        odeRef.R, odeRef.L, odeRef.J, odeRef.b,...
        odeRef.Kt, odeRef.Ke, odeRef.U0,...
        lambdaPhysics, lambdaData);
    
    [net, averageGrad, averageSqGrad] = adamupdate(net, gradients, averageGrad, averageSqGrad, iter, learningRate);
    lossHistoryAdam(iter) = extractdata(loss);

    if mod(iter, 100) == 0 || iter == 1
        disp("Iteration: " + iter + ...
            ", Total Loss: " + extractdata(loss) + ...
            ", Physics Loss: " + extractdata(physicsLoss) + ...
            ", Data Loss: " + extractdata(dataLoss));
    end
end


trainingTimeSeconds = toc(trainingTimer);
fprintf("Training time: %.2f seconds\n", trainingTimeSeconds);

figure

semilogy(1:length(lossHistoryAdam), lossHistoryAdam, "LineWidth",2)
hold on


yPINN = extractdata(predict(net, xInput))';

rmse = sqrt(mean((yPINN(:, 1:2) - yValid).^2, 1));
rmseLoadTorq = sqrt(mean((yPINN(:, 3) - loadValid).^2, 1));

fprintf('RMSE áram:         %.6g A\n', rmse(1));
fprintf('RMSE szögsebesség: %.6g rad/s\n', rmse(2));
fprintf('RMSE terhelés:     %.6g Nm\n', rmseLoadTorq);

figure;

subplot(3,1,1)
plot(tValid, yValid(:,1), 'b-', tValid, yPINN(:,1), 'r--')
ylabel('Áram [A]')
legend('ODE45', 'PINN')
grid on

subplot(3,1,2)
plot(tValid, yValid(:,2), 'b-', tValid, yPINN(:,2), 'r--')
ylabel('Szögsebesség [rad/s]')
legend('ODE45', 'PINN')
grid on

subplot(3,1,3)
plot(tValid, loadValid, 'b-', tValid, yPINN(:,3), 'r--')
xlabel('Idő [s]')
ylabel('Terhelés [Nm]')
legend('ODE45', 'PINN')
grid on

disp("Lefutott minden :)");
