clear;
clc;
close all;

scriptPath = fileparts(mfilename("fullpath"));
addpath(fullfile(fileparts(scriptPath), "Functions"));

%% Physical Parameters
cTrue = 0.25; % True damping coefficient [Ns/m]
kTrue = 0.9; % True spring stiffness [N/m]

cInitial = 0.1; % Initial guess for damping coefficient [Ns/m]
kInitial = 0.5; % Initial guess for spring stiffness [N/m]

m = 1.0; % Mass[kg]
initialDisplacement = 0.1; % Initial displacement [m]
initialVelocity = 0; % Initial velocity [m/s]

%% Time domain

tMax = 30; % Simulation time [s]
numCollocation = 50; % Number of collocation points
t = linspace(0, tMax, numCollocation);
t = dlarray(t, "CB");
t0 = dlarray(0, "CB");

%% Neural Network

layers = [
    featureInputLayer(1)
    fullyConnectedLayer(8)
    tanhLayer
    fullyConnectedLayer(32)
    tanhLayer
    fullyConnectedLayer(32)
    tanhLayer
    fullyConnectedLayer(16)
    tanhLayer
    fullyConnectedLayer(16)
    tanhLayer
    fullyConnectedLayer(1)
];

net = dlnetwork(layers);

%% Noisy data generation
dataPoints = 50;
noiseLevel = 0.02;
[xData, tData] = noisyDataGenerator(tMax, dataPoints, noiseLevel, m, cTrue, kTrue, initialDisplacement, initialVelocity);

%% Training Parameters

numAdamIterations = 30000;
learningRateFirst = 1e-2;
learningRateSecond = 1e-3;
learningrateThird = 1e-3;
learningrateFourth = 1e-3;
learningRateParam = 1e-3;

trailingAvg = [];
trailingAvgSq = [];
avgC = [];
avgSqC = [];
avgK = [];
avgSqK = [];

lambdaPhysics = 50;
lambdaData = 1;

numLBFGSIterations = 200;
lbfgsState = lbfgsState(HistorySize = 20);

lossHistoryAdam = zeros(numAdamIterations, 1);
lossHistoryLBFGS = zeros(numLBFGSIterations, 1);
cHistoryAdam = zeros(numAdamIterations, 1);
kHistoryAdam = zeros(numAdamIterations, 1);

cEst = dlarray(cInitial);
kEst = dlarray(kInitial);

%% Training

trainingTimer = tic;

for iteration = 1:numAdamIterations
    if iteration <= numAdamIterations/4
        learningRate = learningRateFirst;
    elseif iteration <= numAdamIterations/2
        learningRate = learningRateSecond;
    elseif iteration <= 3*numAdamIterations/4
        learningRate = learningrateThird;
    else
        learningRate = learningrateFourth;
    end

    [loss, gradients, physicsLoss, initialConditionLoss, dataLoss, gradC, gradK] = ...
        dlfeval(@modelLoss_LearnableParameters, net, t, t0, tData, xData, m, cEst, kEst, initialDisplacement, initialVelocity, lambdaPhysics, lambdaData);

    [net, trailingAvg, trailingAvgSq] = adamupdate(net, gradients, trailingAvg, trailingAvgSq, iteration, learningRate);
    [cEst, avgC, avgSqC] = adamupdate(cEst, gradC, avgC, avgSqC, iteration, learningRateParam);
    [kEst, avgK, avgSqK] = adamupdate(kEst, gradK, avgK, avgSqK, iteration, learningRateParam);

    cEstimated = extractdata(cEst);
    kEstimated = extractdata(kEst);
    lossHistoryAdam(iteration) = extractdata(loss);
    cHistoryAdam(iteration) = cEstimated;
    kHistoryAdam(iteration) = kEstimated;

    if mod(iteration,100) == 0
        fprintf("Adam Iteration %d\n", iteration);
        fprintf("  Total loss   = %.6e\n", extractdata(loss));
        fprintf("  Physics loss = %.6e\n", extractdata(physicsLoss));
        fprintf("  IC loss      = %.6e\n", extractdata(initialConditionLoss));
        fprintf("  Data loss    = %.6e\n", extractdata(dataLoss));
        fprintf("  c estimate   = %.6f\n", cEstimated);
        fprintf("  k estimate   = %.6f\n", kEstimated);
    end
end

cEstimated = extractdata(cEst);
kEstimated = extractdata(kEst);

lossFcn = @(net) dlfeval(@modelLoss_LearnableParameters, net, t, t0, tData, xData, m, cEst, kEst, initialDisplacement, initialVelocity, lambdaPhysics, lambdaData);

for iteration = 1:numLBFGSIterations
    [net, lbfgsState] = lbfgsupdate(net, lossFcn, lbfgsState);

    [loss, ~, physicsLoss, initialConditionLoss, dataLoss] = ...
        dlfeval(@modelLoss_LearnableParameters, net, t, t0, tData, xData, m, cEst, kEst, initialDisplacement, initialVelocity, lambdaPhysics, lambdaData);

    lossHistoryLBFGS(iteration) = extractdata(loss);

    if mod(iteration, 100) == 0
        fprintf("L-BFGS Iteration %d\n", iteration);
        fprintf("  Total loss   = %.6e\n", extractdata(loss));
        fprintf("  Physics loss = %.6e\n", extractdata(physicsLoss));
        fprintf("  IC loss      = %.6e\n", extractdata(initialConditionLoss));
        fprintf("  Data loss    = %.6e\n", extractdata(dataLoss));
    end
end

trainingTimeSeconds = toc(trainingTimer);
fprintf("Training time: %.2f seconds\n", trainingTimeSeconds);
fprintf("Estimated c = %.6f\n", extractdata(cEst));
fprintf("Estimated k = %.6f\n", extractdata(kEst));

%% Loss plot
lossHistory = [lossHistoryAdam; lossHistoryLBFGS];
figure
semilogy(1:length(lossHistoryAdam), lossHistoryAdam, "LineWidth", 2)
hold on
semilogy(length(lossHistoryAdam)+(1:length(lossHistoryLBFGS)), lossHistoryLBFGS, "LineWidth", 2)
xline(length(lossHistoryAdam), "--", "Adam -> L-BFGS");
grid on
xlabel("Iteration")
ylabel("Loss")
legend("Adam", "L-BFGS")
title("PINN Training Loss")

%% Learnable parameter history
figure

subplot(2, 1, 1)
plot(1:numAdamIterations, cHistoryAdam, "LineWidth", 2)
hold on
yline(cTrue, "--", "True value")
yline(cInitial, ":", "Initial value")
grid on
xlabel("Adam iteration")
ylabel("Damping coefficient c [Ns/m]")
legend("Estimate", "True value", "Initial value")
title("Damping coefficient learning")

subplot(2, 1, 2)
plot(1:numAdamIterations, kHistoryAdam, "LineWidth", 2)
hold on
yline(kTrue, "--", "True value")
yline(kInitial, ":", "Initial value")
grid on
xlabel("Adam iteration")
ylabel("Spring stiffness k [N/m]")
legend("Estimate", "True value", "Initial value")
title("Spring stiffness learning")

%% Test

tTest = linspace(0, tMax, 1000);
tTestDL = dlarray(tTest, "CB");
xPINN = predict(net, tTestDL);
xPINN = extractdata(xPINN);

figure
plot(tTest, xPINN, "LineWidth", 2)
grid on
xlabel("Time [s]")
ylabel("Displacement [m]")
title("PINN prediction")

%% Reference solution using ode45
odeFunction = @(t,x) [
    x(2);
    -(extractdata(cTrue)/m)*x(2) - (extractdata(kTrue)/m)*x(1)
    ];

[tODE, xODE] = ode45(odeFunction, [0 tMax], [initialDisplacement initialVelocity]);

%% Save data for plotting
save(fullfile(scriptPath, "test.mat"), ...
    "net", "lossHistory", "lossHistoryAdam", "lossHistoryLBFGS", ...
    "cHistoryAdam", "kHistoryAdam", ...
    "learningRate", "lambdaPhysics", "lambdaData", ...
    "m", "cEstimated", "kEstimated", "cTrue", "kTrue", "cInitial", "kInitial", ...
    "initialDisplacement", "initialVelocity", ...
    "numCollocation", "tMax", "trainingTimeSeconds", ...
    "tData", "xData", "tTest", "xPINN", "tODE", "xODE");

figure
plot(tODE, xODE(:,1), "LineWidth", 2)
hold on
plot(tTest, xPINN, "--", "LineWidth", 2)
grid on
xlabel("Time [s]")
ylabel("Displacement [m]")
legend("ode45", "PINN")
title("Spring-Mass-Damper: PINN vs. ode45")

%% PINN vs Data points
figure
plot(tData, xData, "o", "LineWidth", 2)
hold on
plot(tTest, xPINN, "LineWidth", 2)
grid on
xlabel("Time [s]")
ylabel("Displacement [m]")
legend("Data points", "PINN")
title("Spring-Mass-Damper: PINN vs. Data points")
