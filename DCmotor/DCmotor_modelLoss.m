function [loss, gradients, physicsLoss, initialConditionLoss, dataLoss] = ...
    DCmotor_modelLoss(net, t, t0, tData, yData,...
    R, L, J, b, Kt, Ke, U0, load_torque, y0,...
    lambdaPhysics, lambdaData)

    %% Derivatives
    [x, dxdt] = DCmotor_modelDerivatives(net, t);
     
    %% Physics loss
    resI = L * dxdt(1, :) - (U0 - R * x(1, :) -Ke * x(2, :));
    resOmega = J * dxdt(2, :) - (Kt * x(1, :) - b * x(2, :) - load_torque);
    resPos = dxdt(3, :) - x(2, :);

    physicsLoss = mean(resI.^2 + resOmega.^2 + resPos.^2);
    
    %% Data loss
    xDataPred = forward(net, tData);
    dataLoss = mean((xDataPred - yData).^2, "all");

    %% Initial Condition Loss
    
    [x0, ~] = DCmotor_initialConditionDerivative(net, t0);
    initialConditionLoss = mean((x0 - y0).^2);
    
    %% Total Loss
    
    loss = lambdaPhysics * (physicsLoss + initialConditionLoss) + lambdaData * dataLoss;
    
    %% Gradients
    
    gradients = dlgradient(loss, net.Learnables);
end
