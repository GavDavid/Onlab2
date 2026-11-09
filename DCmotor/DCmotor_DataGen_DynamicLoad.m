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

y0 = [0; 0; 0]; % initial state: [current; angular velocity; angular position]
t = linspace(0, 10, 1000); 

load_torque = @(t) 0 * (t < 2.5) + 0.1 * (t >= 2.5 & t < 5) + 0.1 * (t >= 5 & t < 7.5) + 0 * (t >= 7.5);

% State equations
eq = @(t, y) [(U0 -R*y(1) -Ke*y(2))/L;
              (Kt*y(1) - b*y(2) - load_torque(t))/J;
              y(2)];

[t, y] = ode45(eq, t, y0);

load_torque = load_torque(t);

save('DCmotor_ODE45_reference_DynamicLoad.mat', 't', 'y', 'R', 'L', 'J', 'b', 'Kt', 'Ke', 'U0', 'load_torque');


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
plot(t, load_torque, 'LineWidth', 1.5)
xlabel('Idő [s]')
ylabel('Terhelés [Nm]')
grid on