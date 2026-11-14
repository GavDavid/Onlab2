clear;
clc;
clf;
close all;


odeRef = load("DCmotor_ODE45_reference_DynamicLoad.mat");
dt = odeRef.t(2) - odeRef.t(1);

noiseLevel = 0.05;
iMeas = odeRef.y(:, 1)' + noiseLevel * randn(size(odeRef.y(:, 1)'));
omegaMeas = odeRef.y(:, 2)' + noiseLevel * randn(size(odeRef.y(:, 2)'));
U = odeRef.U0 * ones(size(iMeas));


xInput = [
    U(1:end);         [U(1), U(1:end-1)];      [U(1), U(1), U(1:end-2)];
    iMeas(1:end);     [0, iMeas(1:end-1)];     [0, 0, iMeas(1:end-2)];
    omegaMeas(1:end); [0, omegaMeas(1:end-1)]; [0, 0, omegaMeas(1:end-2)]
];

xInput = dlarray(xInput, "CB");
yData = dlarray([iMeas; omegaMeas], "CB");


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
lambdaPhysics = 5;
lambdaData = 1;

% Adam optimizer parameters
numAdamIterations = 20000;
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

    if mod(iter, 500) == 0 || iter == 1
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


dOmegaMeas = (omegaMeas - [0, omegaMeas(1:end-1)]) / dt;
loadEq = (odeRef.Kt * iMeas - odeRef.b * omegaMeas - odeRef.J * dOmegaMeas)';

rmse = sqrt(mean((yPINN(:, 1:2) - odeRef.y(:, 1:2)).^2, 1));
rmseLoadTorq = sqrt(mean((yPINN(:, 3) - odeRef.load_torque).^2, 1));
rmseLoadEq = sqrt(mean((loadEq - odeRef.load_torque).^2, 1));

fprintf('RMSE áram:                %.6g A\n', rmse(1));
fprintf('RMSE szögsebesség:        %.6g rad/s\n', rmse(2));
fprintf('RMSE terhelés (PINN):     %.6g Nm\n', rmseLoadTorq);
fprintf('RMSE terhelés (Egyenlet): %.6g Nm\n', rmseLoadEq);

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
plot(odeRef.t, loadEq, 'g-', odeRef.t, odeRef.load_torque, 'b-', odeRef.t, yPINN(:,3), 'r--', 'LineWidth', 1.2)
xlabel('Idő [s]')
ylabel('Terhelés [Nm]')
legend('Egyenlet (zajos)', 'ODE45', 'PINN')
grid on


%% Tesztelés nem látott terhelésprofilon
load_torque_test = @(t) 0 * (t < 1.5) + 0.05 * (t >= 1.5 & t < 4.5) + 0.12 * (t >= 4.5 & t < 8.0) + 0 * (t >= 8.0);

eqTest = @(t, y) [(odeRef.U0 - odeRef.R*y(1) - odeRef.Ke*y(2))/odeRef.L;
                  (odeRef.Kt*y(1) - odeRef.b*y(2) - load_torque_test(t))/odeRef.J;
                  y(2)];

[tTest, yTest] = ode45(eqTest, odeRef.t, [0; 0; 0]);
loadTest = load_torque_test(tTest);

iMeasTest = yTest(:, 1)' + noiseLevel * randn(size(yTest(:, 1)'));
omegaMeasTest = yTest(:, 2)' + noiseLevel * randn(size(yTest(:, 2)'));

xInputTest = [
    U(1:end);             [U(1), U(1:end-1)];          [U(1), U(1), U(1:end-2)];
    iMeasTest(1:end);     [0, iMeasTest(1:end-1)];     [0, 0, iMeasTest(1:end-2)];
    omegaMeasTest(1:end); [0, omegaMeasTest(1:end-1)]; [0, 0, omegaMeasTest(1:end-2)]
];
xInputTest = dlarray(xInputTest, "CB");

yPINNTest = extractdata(predict(net, xInputTest))';

dOmegaMeasTest = (omegaMeasTest - [0, omegaMeasTest(1:end-1)]) / dt;
loadEqTest = (odeRef.Kt * iMeasTest - odeRef.b * omegaMeasTest - odeRef.J * dOmegaMeasTest)';

rmseTest = sqrt(mean((yPINNTest(:, 1:2) - yTest(:, 1:2)).^2, 1));
rmseLoadTorqTest = sqrt(mean((yPINNTest(:, 3) - loadTest).^2, 1));
rmseLoadEqTest = sqrt(mean((loadEqTest - loadTest).^2, 1));

fprintf('\n--- TESZT (nem látott terhelésprofil) ---\n');
fprintf('RMSE áram (Teszt):                %.6g A\n', rmseTest(1));
fprintf('RMSE szögsebesség (Teszt):        %.6g rad/s\n', rmseTest(2));
fprintf('RMSE terhelés (PINN - Teszt):     %.6g Nm\n', rmseLoadTorqTest);
fprintf('RMSE terhelés (Egyenlet - Teszt): %.6g Nm\n', rmseLoadEqTest);

figure;

subplot(3,1,1)
plot(tTest, yTest(:,1), 'b-', tTest, yPINNTest(:,1), 'r--')
title('Tesztelés nem látott terhelésprofilon')
ylabel('Áram [A]')
legend('ODE45', 'PINN')
grid on

subplot(3,1,2)
plot(tTest, yTest(:,2), 'b-', tTest, yPINNTest(:,2), 'r--')
ylabel('Szögsebesség [rad/s]')
legend('ODE45', 'PINN')
grid on

subplot(3,1,3)
plot(tTest, loadEqTest, 'g-', tTest, loadTest, 'b-', tTest, yPINNTest(:,3), 'r--', 'LineWidth', 1.2)
xlabel('Idő [s]')
ylabel('Terhelés [Nm]')
legend('Egyenlet (zajos)', 'ODE45', 'PINN')
grid on

disp("Lefutott minden :)");
