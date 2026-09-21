function [x,dxdt,d2xdt2] = modelDerivatives(net,t)

    x = forward(net,t);

    dxdt = dlgradient(sum(x,"all"), t, EnableHigherDerivatives=true);

    d2xdt2 = dlgradient(sum(dxdt,"all"), t, EnableHigherDerivatives=true);

end