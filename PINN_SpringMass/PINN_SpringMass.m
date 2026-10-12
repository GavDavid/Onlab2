clear;
clc;
close all;

scriptPath = fileparts(mfilename("fullpath"));
addpath(fullfile(fileparts(scriptPath), "Functions"));

%% Physical Parameters
 
m = 1.0; % Mass[kg]
c = 0.25; % Damping coefficient [Ns/m]
k = 1.0; % Spring stiffness [N/m]
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
    fullyConnectedLayer(25)
    tanhLayer
    fullyConnectedLayer(25)
    tanhLayer
    fullyConnectedLayer(25)
    tanhLayer
    fullyConnectedLayer(25)
    tanhLayer
    fullyConnectedLayer(25)
    tanhLayer
    fullyConnectedLayer(1)
];

net = dlnetwork(layers);

%% Noisy data generation
dataPoints = 50;
noiseLevel = 0.02;
[xData, tData] = noisyDataGenerator(tMax, dataPoints, noiseLevel, m, c, k, initialDisplacement, initialVelocity);

%% Training Parameters

numAdamIterations = 20000;
learningRateFirst = 1e-2;
learningRateSecond = 1e-3;
learningrateThird = 1e-4;
learningrateFourth = 1e-5;
trailingAvg = [];
trailingAvgSq = [];
lambdaPhysics = 1;
lambdaData = 1;

numLBFGSIterations = 100;
lbfgsState = lbfgsState(HistorySize = 20);

lossHistoryAdam = zeros(numAdamIterations, 1);
lossHistoryLBFGS = zeros(numLBFGSIterations, 1);



 
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
    [loss, gradients, physicsLoss, initialConditionLoss, dataLoss] = ...
    dlfeval(@modelLoss, net, t, t0, tData, xData, m, c, k, initialDisplacement, initialVelocity, lambdaPhysics, lambdaData);

    [net, trailingAvg, trailingAvgSq] = adamupdate(net, gradients, trailingAvg, trailingAvgSq, iteration, learningRate);


    lossHistoryAdam(iteration) = extractdata(loss);

    if mod(iteration,100) == 0

        fprintf("Adam Iteration %d\n", iteration);
        fprintf("  Total loss   = %.6e\n", extractdata(loss));
        fprintf("  Physics loss = %.6e\n", extractdata(physicsLoss));
        fprintf("  IC loss      = %.6e\n", extractdata(initialConditionLoss));
        fprintf("  Data loss      = %.6e\n", extractdata(dataLoss));

    end

end

lossFcn = @(net) dlfeval(@modelLoss, net, t, t0, tData, xData, m, c, k, initialDisplacement, initialVelocity, lambdaPhysics, lambdaData);

for iteration = 1:numLBFGSIterations

    [net, lbfgsState] = lbfgsupdate(net, lossFcn, lbfgsState);


    [loss, ~, physicsLoss, initialConditionLoss, dataLoss] = dlfeval(@modelLoss, net, t, t0, tData, xData, m, c, k, initialDisplacement, initialVelocity, lambdaPhysics, lambdaData);

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


%% Loss plot


lossHistory = [lossHistoryAdam; lossHistoryLBFGS];

figure

semilogy(1:length(lossHistoryAdam), ...
    lossHistoryAdam, ...
    "LineWidth",2)

hold on

semilogy( ...
    length(lossHistoryAdam)+(1:length(lossHistoryLBFGS)), ...
    lossHistoryLBFGS, ...
    "LineWidth",2)

xline(length(lossHistoryAdam), "--", "Adam -> L-BFGS");

grid on

xlabel("Iteration")
ylabel("Loss")
legend("Adam","L-BFGS")

title("PINN Training Loss")

%% Test

tTest = linspace(0, tMax, 1000);
tTestDL = dlarray(tTest, "CB");

xPINN = predict(net, tTestDL);
xPINN = extractdata(xPINN);

figure

plot(tTest,xPINN,"LineWidth",2)

grid on

xlabel("Time [s]")
ylabel("Displacement [m]")

title("PINN prediction")

%% Reference solution using ode45

odeFunction = @(t,x) [
    x(2);
    -(c/m)*x(2) - (k/m)*x(1)
    ];

[tODE,xODE] = ode45( odeFunction, [0 tMax], [initialDisplacement initialVelocity]);

%% Save data for plotting

save(fullfile(scriptPath, "lepcsosADAMwithLBFGS.mat"), ...
    "net", "lossHistory", "lossHistoryAdam", "lossHistoryLBFGS", ...
    "learningRate", "lambdaPhysics", "lambdaData", ...
    "m", "c", "k", "initialDisplacement", "initialVelocity", ...
    "numCollocation", "tMax", "trainingTimeSeconds", ...
    "tData", "xData", "tTest", "xPINN", "tODE", "xODE");

figure

plot(tODE,xODE(:,1),"LineWidth",2)
hold on

plot(tTest,xPINN,"--","LineWidth",2)

grid on

xlabel("Time [s]")
ylabel("Displacement [m]")

legend("ode45","PINN")

title("Spring-Mass-Damper: PINN vs. ode45")

%% PINN vs Data points
figure

plot(tData, xData,"o", "LineWidth",2)
hold on

plot(tTest,xPINN,"LineWidth",2)

grid on

xlabel("Time [s]")
ylabel("Displacement [m]")

legend("Data points","PINN")

title("Spring-Mass-Damper: PINN vs. Data points")
