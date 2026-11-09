function [loss, gradients, physicsLoss, initialConditionLoss, dataLoss, loadChangeLoss] = ...
    DCmotor_modelLoss_DynamicLoad(net, t, t0, tData, yData,...
    R, L, J, b, Kt, Ke, U, y0,...
    lambdaPhysics, lambdaData, lambdaLoadChange)

    %% Derivatives
    [x, dxdt] = DCmotor_modelDerivatives(net, t);
     
    %% Physics loss
    loadTorquePred = x(4,:);

    resI = L * dxdt(1, :) - (U - R * x(1, :) -Ke * x(2, :));
    resOmega = J * dxdt(2, :) - (Kt * x(1, :) - b * x(2, :) - loadTorquePred);
    resPos = dxdt(3, :) - x(2, :);

    physicsLoss = mean(resI.^2 + resOmega.^2 + resPos.^2);
    
    %% Data loss
    xDataPred = forward(net, tData);
    dataLoss = mean((xDataPred(1:2, :) - yData).^2, "all");

    %% Initial Condition Loss
    
    [x0, ~] = DCmotor_initialConditionDerivative(net, t0);
    initialConditionLoss = mean((x0(1:3, :) - y0).^2);
    
    %% Load Change Loss
    loadPrediction = x(4, :);
    deltaLoad = loadPrediction(2:end) - loadPrediction(1:end-1);
    epsChange = 1e-6; 
    loadChangeLoss = sum((deltaLoad.^2 + epsChange^2).^0.25 - epsChange^0.5);

    %% Total Loss
    
    loss = lambdaPhysics * (physicsLoss + initialConditionLoss) + lambdaData * dataLoss + lambdaLoadChange * loadChangeLoss;
    
    %% Gradients
    
    [gradients] = dlgradient(loss, net.Learnables);
end
