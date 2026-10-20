clear;
clc;
close all;

%% Load saved data

scriptPath = fileparts(mfilename("fullpath"));
dataFile = fullfile(scriptPath, "learnedParamsHighNoise4.mat");

if ~isfile(dataFile)
    error("PINN_SpringMass_LearnableParameters_Plotter:MissingDataFile", ...
        "The data file '%s' was not found. Run PINN_SpringMass_LearnableParameters.m first.", dataFile);
end

data = load(dataFile);
requiredFields = {'lossHistoryAdam', 'lossHistoryLBFGS', ...
    'cTrue', 'kTrue', 'cEstimated', 'kEstimated', ...
    'tData', 'xData', 'tTest', 'xPINN', ...
    'tODE', 'xODE', 'tMax', 'trainingTimeSeconds'};
missingFields = requiredFields(~isfield(data, requiredFields));

if ~isempty(missingFields)
    error("PINN_SpringMass_LearnableParameters_Plotter:IncompleteDataFile", ...
        "The data file is missing required variables: %s. Run PINN_SpringMass_LearnableParameters.m to regenerate it.", ...
        strjoin(missingFields, ', '));
end

fprintf("Training time: %.2f seconds\n", data.trainingTimeSeconds);
fprintf("Estimated c = %.6f\n", data.cEstimated);
fprintf("Estimated k = %.6f\n", data.kEstimated);

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

%% Learnable parameter history

hasParameterHistory = isfield(data, 'cHistoryAdam') && isfield(data, 'kHistoryAdam');

if hasParameterHistory
    figure

    subplot(2, 1, 1)
    plot(1:length(data.cHistoryAdam), data.cHistoryAdam, "LineWidth", 2)
    hold on
    yline(data.cTrue, "--", "True value")
    if isfield(data, 'cInitial')
        yline(data.cInitial, ":", "Initial value")
        legend("Estimate", "True value", "Initial value")
    else
        legend("Estimate", "True value")
    end
    grid on
    xlabel("Adam iteration")
    ylabel("Damping coefficient c [Ns/m]")
    title("Damping coefficient learning")

    subplot(2, 1, 2)
    plot(1:length(data.kHistoryAdam), data.kHistoryAdam, "LineWidth", 2)
    hold on
    yline(data.kTrue, "--", "True value")
    if isfield(data, 'kInitial')
        yline(data.kInitial, ":", "Initial value")
        legend("Estimate", "True value", "Initial value")
    else
        legend("Estimate", "True value")
    end
    grid on
    xlabel("Adam iteration")
    ylabel("Spring stiffness k [N/m]")
    title("Spring stiffness learning")
else
    warning("PINN_SpringMass_LearnableParameters_Plotter:MissingParameterHistory", ...
        "The saved data does not contain cHistoryAdam and kHistoryAdam. Skipping the parameter-history plot; rerun PINN_SpringMass_LearnableParameters.m to save it.");
end

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
