function pep = compute_PEP(out)
%% PEP = COMPUTE_PEP(OUT)
%
% computes protected exceedance probability (Rigoux et al., 2014)
% for rfx-bms
%
% Timo Flesch, 2018

pep = (1-out.bor)*out.ep + out.bor/length(out.ep);
end
