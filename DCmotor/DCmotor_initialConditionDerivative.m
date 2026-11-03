function [x0, dxdt0] = DCmotor_initialConditionDerivative(net, t0)
    
    [x0, dxdt0] = DCmotor_modelDerivatives(net, t0);

end