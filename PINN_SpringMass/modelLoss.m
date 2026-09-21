function [loss, gradients, physicsLoss, initialConditionLoss, dataLoss] = ...
modelLoss(net, t, t0, tData, xData, m, c, k, initialDisplacement, initialVelocity, lambdaPhysics, lambdaData)

    %% Derivatives
    
    [x, dxdt, d2xdt2] = modelDerivatives(net, t);
     
    %% Physics loss
    
    residual = m * d2xdt2 + c * dxdt + k * x;
    physicsLoss = mean(residual.^2);
    
    %% Data loss
    xDataPred = forward(net, tData);
    dataLoss = mean((xDataPred - xData).^2 );

    %% Initial Condition Loss
    
    [x0, dxdt0] = initialConditionDerivative(net, t0);
    initialConditionLoss = (x0 - initialDisplacement).^2 + (dxdt0 - initialVelocity).^2;
    
    %% Total Loss
    
    loss = lambdaPhysics * (physicsLoss + initialConditionLoss) + lambdaData * dataLoss;
    
    %% Gradients
    
    gradients = dlgradient(loss, net.Learnables);
end
