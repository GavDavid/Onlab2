clear;
clc;
clf;
close all;

R = 2.0; % armature resistance [ohms]
L = 0.5; % armature inductance [H]
J = 0.02; % rotor inertia [kg*m^2]
b = 0.2; % viscous friction coefficient [Nm*s]
Kt = 0.1; % motor torque constant [Nm/A]
Ke = 0.1; % back EMF constant [Vs/rad]
U0 = 6; % applied voltage [V]
U1 = 12; % applied voltage [V]
U2 = 4; % applied voltage [V]
U3 = 9; % applied voltage [V]
load_torque = 0; % load torque [Nm]

y0 = [0; 0; 0]; % initial state: [current; angular velocity; angular position]
t = linspace(0, 10, 1000); 

U = @(t) U0 * (t < 2.5) + U1 * (t >= 2.5 & t < 5) + U2 * (t >= 5 & t < 7.5) + U3 * (t >= 7.5);

% State equations
eq = @(t, y) [(U(t) -R*y(1) -Ke*y(2))/L;
              (Kt*y(1) - b*y(2) - load_torque)/J;
              y(2)];

[t, y] = ode45(eq, t, y0);

save('DCmotor_ODE45_reference_ChangingVoltage.mat', 't', 'y', 'R', 'L', 'J', 'b', 'Kt', 'Ke', 'U', 'load_torque');


figure;

subplot(4,1,1)
plot(t, y(:,1))
ylabel('Áram [A]')
grid on

subplot(4,1,2)
plot(t, y(:,2))
ylabel('Szögsebesség [rad/s]')
grid on

subplot(4,1,3)
plot(t, y(:,3))
ylabel('Pozíció [rad]')
xlabel('Idő [s]')
grid on

subplot(4,1,4)
plot(t, U(t), 'LineWidth', 1.5)
xlabel('Idő [s]')
ylabel('Feszültség [V]')
grid on