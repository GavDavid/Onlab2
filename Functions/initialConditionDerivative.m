function [x0,dxdt0] = initialConditionDerivative(net, t0)
    
    x0 = forward(net, t0);
    
    dxdt0 = dlgradient(sum(x0, "all"), t0, EnableHigherDerivatives=true);

end