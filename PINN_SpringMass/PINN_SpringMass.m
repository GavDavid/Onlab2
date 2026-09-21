clear;
clc;
close all;

%% Physical Parameters

m = 1.0; % Mass[kg]
c = 0.5; % Damping coefficient [Ns/m]
k = 2.0; % Spring stiffness [N/m]
initialDisplacement = 0.1; % Initial displacement [m]
initialVelocity = 0; % Initial velocity [m/s]

%% Time domain

tMax = 30; % Simulation time [s]
numCollocation = 1000; % Number of collocation points
t = linspace(0, tMax, numCollocation);
t = dlarray(t, "CB");
t0 = dlarray(0, "CB");

%% Neural Network

layers = [
    featureInputLayer(1)
    fullyConnectedLayer(32)
    tanhLayer
    fullyConnectedLayer(32)
    tanhLayer
    fullyConnectedLayer(32)
    tanhLayer
    fullyConnectedLayer(1)
];

net = dlnetwork(layers);

%% Noisy data generation
[xData, tData] = noisyDataGenerator(tMax, 50, 0.002, m, c, k, initialDisplacement, initialVelocity);

%% Training Parameters

numIterations = 1000;
learningRate = 1e-3;
trailingAvg = [];
trailingAvgSq = [];
lossHistory = zeros(numIterations, 1);
lambdaPhysics = 1;
lambdaData = 0;
%% Training

for iteration = 1:numIterations

    [loss, gradients, physicsLoss, initialConditionLoss, dataLoss] = ...
    dlfeval(@modelLoss, net, t, t0, tData, xData, m, c, k, initialDisplacement, initialVelocity, lambdaPhysics, lambdaData);

    [net, trailingAvg, trailingAvgSq] = adamupdate(net, gradients, trailingAvg, trailingAvgSq, iteration, learningRate);

    lossHistory(iteration) = extractdata(loss);

    if mod(iteration,100) == 0

        fprintf("Iteration %d\n", iteration);
        fprintf("  Total loss   = %.6e\n", extractdata(loss));
        fprintf("  Physics loss = %.6e\n", extractdata(physicsLoss));
        fprintf("  IC loss      = %.6e\n", extractdata(initialConditionLoss));
        fprintf("  Data loss      = %.6e\n", extractdata(dataLoss));

    end

end

figure

semilogy(lossHistory)

grid on

xlabel("Iteration")
ylabel("Loss")
title("PINN training")

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
