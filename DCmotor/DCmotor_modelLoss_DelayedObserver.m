function [loss, gradients, physicsLoss, dataLoss] = ...
    DCmotor_modelLoss_DelayedObserver(net, xInput, yData, dt,...
    R, L, J, b, Kt, Ke, U,...
    lambdaPhysics, lambdaData)

    %% Prediction and Derivatives
    x = forward(net, xInput);
    dxdt = (x - [zeros(3, 1), x(:, 1:end-1)]) / dt;
     
    %% Physics loss
    iPred = x(1, 1:end);
    omegaPred = x(2, 1:end);
    loadTorquePred = x(3, 1:end);

    resI = L * dxdt(1, :) - (U - R * iPred - Ke * omegaPred);
    resOmega = J * dxdt(2, :) - (Kt * iPred - b * omegaPred - loadTorquePred);
    resLoad = 0.005 * dxdt(3, :);

    physicsLoss = mean(resI.^2 + resOmega.^2 + resLoad.^2);
    
    %% Data loss
    dataLoss = mean((x(1:2, :) - yData).^2, "all");

    %% Total Loss
    loss = lambdaPhysics * physicsLoss + lambdaData * dataLoss;
    
    %% Gradients
    [gradients] = dlgradient(loss, net.Learnables);
end
