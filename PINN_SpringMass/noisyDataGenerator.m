function [xMeas, tMeas] = noisyDataGenerator(tMax,dataNum, noiseLevel, m, c, k, initialDisplacement, initialVelocity)

tMeas = linspace(0, tMax, dataNum);

odeFunction = @(t,x) [
    x(2);
    -(c/m)*x(2) - (k/m)*x(1)
    ];

[tMeas, xTrue] = ode45(odeFunction, tMeas, [initialDisplacement, initialVelocity]);

xMeas = xTrue(:,1) + noiseLevel * randn(size(xTrue(:,1)));
xMeas = dlarray(xMeas', "CB");
tMeas = dlarray(tMeas', "CB");

end