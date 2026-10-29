function DeltaE_bit_pJ = calc_extra_energy(PAMorder, baudrate, dE_dB, Pin_dBm)
% --------------------
% Calculate the additional energy consumption (in pJ/bit) required to compensate
% for the insertion loss of an optical computing module.
%
% Inputs:
%   PAMorder   : Modulation format
%   baudrate   : Baud rate 
%   dE_dB      : Insertion loss of the optical computing module (dB)
%   Pin_dBm    : Original input optical power (dBm)
%
% Output:
%   DeltaE_bit_pJ : Additional power consumption per bit (pJ/bit)
% -------------------------------------------------------------

    bits_per_symbol = log2(PAMorder);
    Rb = baudrate * bits_per_symbol;   % bit/s

    Pin_W = 1e-3 * 10^(Pin_dBm/10);
    loss_factor = 10^(dE_dB/10);

    deltaP_W = Pin_W * (loss_factor - 1);
    DeltaE_bit_pJ = (deltaP_W / Rb) * 1e12;
end
