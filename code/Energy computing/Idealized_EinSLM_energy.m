% clc;
clear;
close all;

%%
PAMorder=4;
n=1.44;
c = 3e8/n;
G = 1e9;
lambda = 1550e-9;
w0 = 5e-5;
layer = 4;

zl = 1e-3;
zl_num = zl * 1e3;

target_delta = 1e-6;
delta = 1e-6;
N = 2400;
L = (N)*delta;

mu_factor = target_delta/delta;
out_delta = target_delta;

N_fall = 130;
bordsize_fall = N/2-N_fall*mu_factor/2;
Size_fall = [N_fall, N_fall];

load('Idealized_26G.mat', 'data');

EinSLM = data.EinSLM;
E1 = data.E1;
baudrate = data.baudrate;
bitrate = baudrate * log2(PAMorder);
fprintf('OSTU field loaded\n');
%% 

P_target = 0;                            
P_current = complexFieldPower_dBm(E1, delta);
amp = 10^((P_target - P_current)/20);
E1 = E1 * amp;

Pin_dBm = complexFieldPower_dBm(E1, delta); 
EinSLM = amp*EinSLM;
P_o = complexFieldPower_dBm(EinSLM, target_delta);

dE = Pin_dBm-P_o;
disp([' E1 Power is ','*',num2str(Pin_dBm),'dbm*'])
disp([' ** Eout Power is ','*',num2str(P_o),'dbm*'])
disp([' dE Power is ','*',num2str(dE),'dbm*'])

eltaO_bit_pJ = calc_extra_energy(PAMorder, baudrate, dE, Pin_dBm);
wall_plug_eff = 0.1;
eltaE_bit_pJ = eltaO_bit_pJ/0.1;

disp([' eltaO_bit_pJ is ','*',num2str(eltaO_bit_pJ),'pJ/bit'])
disp([' eltaE_bit_pJ is ','*',num2str(eltaE_bit_pJ),'pJ/bit'])

fprintf('%g\t%g\t%g\t%g\t%g\n',baudrate/G,bitrate/G,dE,eltaO_bit_pJ,eltaE_bit_pJ);