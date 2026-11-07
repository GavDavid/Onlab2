# Ez a mérés arra vonatkozik, hogy a data pointok száma hogyan befolyásolja a paraméterek becslését
lambdaPhysics = 1;
lambdaData = 1;
numCollocation = 50;
numAdamIterations = 10000;
learningRate = 1e-3;
net: 1x16x16x16x3 (tanh)


- 50 adatpont
RMSE áram:         0.000830657 A
RMSE szögsebesség: 0.00669897 rad/s
RMSE pozíció:      0.00100759 rad
J becslés: 0.0200137 kg*m^2, igaz érték: 0.02 kg*m^2
b becslés: 0.200106 Nm*s, igaz érték: 0.2 Nm*s

- 25 adatpont
RMSE áram:         0.00106531 A
RMSE szögsebesség: 0.0107291 rad/s
RMSE pozíció:      0.00128201 rad
J becslés: 0.0198515 kg*m^2, igaz érték: 0.02 kg*m^2
b becslés: 0.200115 Nm*s, igaz érték: 0.2 Nm*s

- 13 adatpont
RMSE áram:         0.000972151 A
RMSE szögsebesség: 0.0052498 rad/s
RMSE pozíció:      0.00128745 rad
J becslés: 0.0197141 kg*m^2, igaz érték: 0.02 kg*m^2
b becslés: 0.200138 Nm*s, igaz érték: 0.2 Nm*s

- 5 adatpont
RMSE áram:         0.0042896 A
RMSE szögsebesség: 0.0246689 rad/s
RMSE pozíció:      0.00516293 rad
J becslés: 0.0185031 kg*m^2, igaz érték: 0.02 kg*m^2
b becslés: 0.200736 Nm*s, igaz érték: 0.2 Nm*s

- 3 adatpont
RMSE áram:         0.00178916 A
RMSE szögsebesség: 0.0278136 rad/s
RMSE pozíció:      0.00632348 rad
J becslés: 0.0198949 kg*m^2, igaz érték: 0.02 kg*m^2
b becslés: 0.199858 Nm*s, igaz érték: 0.2 Nm*s

- 1 adatpont
RMSE áram:         0.00727637 A
RMSE szögsebesség: 0.20013 rad/s
RMSE pozíció:      0.096949 rad
J becslés: 0.00207229 kg*m^2, igaz érték: 0.02 kg*m^2
b becslés: 0.210224 Nm*s, igaz érték: 0.2 Nm*s

- 3 adatpont v2
RMSE áram:         0.00149318 A
RMSE szögsebesség: 0.0221532 rad/s
RMSE pozíció:      0.00708672 rad
J becslés: 0.0205501 kg*m^2, igaz érték: 0.02 kg*m^2
b becslés: 0.199577 Nm*s, igaz érték: 0.2 Nm*s
