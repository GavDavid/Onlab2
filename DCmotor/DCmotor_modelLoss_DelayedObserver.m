function [loss, gradients, physicsLoss, dataLoss] = ...
    DCmotor_modelLoss_DelayedObserver(net, xInput, yData, dt,...
    R, L, J, b, Kt, Ke, U,...
    lambdaPhysics, lambdaData)

    %% Prediction and Derivatives
    x = forward(net, xInput);
    dxdt = (x(:, 2:end) - x(:, 1:end-1)) / dt;
     
    %% Physics loss
    iPred = x(1, 2:end);
    omegaPred = x(2, 2:end);
    loadTorquePred = x(3, 2:end);

    resI = L * dxdt(1, :) - (U - R * iPred - Ke * omegaPred);
    resOmega = J * dxdt(2, :) - (Kt * iPred - b * omegaPred - loadTorquePred);

    physicsLoss = mean(resI.^2 + resOmega.^2);
    
    %% Data loss
    dataLoss = mean((x(1:2, :) - yData).^2, "all");

    %% Total Loss
    loss = lambdaPhysics * physicsLoss + lambdaData * dataLoss;
    
    %% Gradients
    [gradients] = dlgradient(loss, net.Learnables);
end
