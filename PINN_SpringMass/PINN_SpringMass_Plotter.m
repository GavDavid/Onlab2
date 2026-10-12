clear;
clc;
close all;

%% Load saved data

scriptPath = fileparts(mfilename("fullpath"));
dataFile = fullfile(scriptPath, "ADAMandLBFGS.mat");

if ~isfile(dataFile)
    error("PINN_SpringMass_Plotter:MissingDataFile", ...
        "The data file '%s' was not found. Run PINN_SpringMass.m first.", dataFile);
end

data = load(dataFile);
requiredFields = {'lossHistoryAdam', 'lossHistoryLBFGS', ...
    'tData', 'xData', 'tTest', 'xPINN', 'tODE', 'xODE', 'tMax', ...
    'trainingTimeSeconds'};
missingFields = requiredFields(~isfield(data, requiredFields));

if ~isempty(missingFields)
    error("PINN_SpringMass_Plotter:IncompleteDataFile", ...
        "The data file is missing required variables: %s. Run PINN_SpringMass.m to regenerate it.", ...
        strjoin(missingFields, ', '));
end

fprintf("Training time: %.2f seconds\n", data.trainingTimeSeconds);

%% Loss plot

figure

semilogy(1:length(data.lossHistoryAdam), ...
    data.lossHistoryAdam, ...
    "LineWidth",2)

hold on

semilogy( ...
    length(data.lossHistoryAdam)+(1:length(data.lossHistoryLBFGS)), ...
    data.lossHistoryLBFGS, ...
    "LineWidth",2)

xline(length(data.lossHistoryAdam), "--", "Adam -> L-BFGS");

grid on

xlabel("Iteration")
ylabel("Loss")
legend("Adam","L-BFGS")

title("PINN Training Loss")

%% PINN prediction

figure

plot(data.tTest, data.xPINN, "LineWidth",2)

grid on

xlabel("Time [s]")
ylabel("Displacement [m]")

title("PINN prediction")

%% Reference solution using ode45

figure

plot(data.tODE, data.xODE(:,1), "LineWidth",2)
hold on

plot(data.tTest, data.xPINN, "--", "LineWidth",2)

grid on

xlabel("Time [s]")
ylabel("Displacement [m]")

legend("ode45","PINN")

title("Spring-Mass-Damper: PINN vs. ode45")

%% PINN vs Data points

figure

plot(data.tData, data.xData, "o", "LineWidth",2)
hold on

plot(data.tTest, data.xPINN, "LineWidth",2)

grid on

xlabel("Time [s]")
ylabel("Displacement [m]")

legend("Data points","PINN")

title("Spring-Mass-Damper: PINN vs. Data points")
