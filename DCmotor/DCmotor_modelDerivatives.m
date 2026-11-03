function [x,dxdt] = DCmotor_modelDerivatives(net,t)

    x = forward(net,t);

    didt = dlgradient(sum(x(1, :),"all"), t, EnableHigherDerivatives=true);
    domega_dt = dlgradient(sum(x(2, :),"all"), t, EnableHigherDerivatives=true);
    dtheta_dt = dlgradient(sum(x(3, :),"all"), t, EnableHigherDerivatives=true);

    dxdt = [didt; domega_dt; dtheta_dt];

end