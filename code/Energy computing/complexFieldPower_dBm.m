function [P_dBm_total] = complexFieldPower_dBm(E, delta)
% --------------------------------------------------------
% Calculate the total power (in dBm) of an entire NxN optical field.
%
% Inputs:
%   E        : NxN complex-valued optical field matrix.
%   delta    : Spatial sampling interval [m].
%
% Outputs:
%   P_dBm_total  : Total power of the entire plane (in dBm).
%
% --------------------------------------------------------

    I = abs(E).^2;

    P_total = sum(I(:)) * (delta^2);
    P_dBm_total = 10 * log10(P_total / 1e-3);
end
