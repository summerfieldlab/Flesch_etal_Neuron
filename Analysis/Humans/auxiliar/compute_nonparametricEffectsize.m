function r = compute_nonparametricEffectsize(Z,N)
%% R = COMPUTE_NONPARAMETRICEFFECTSIZE(Z,N)
%
% computes effect size of nonparametric test (signrank,ranksum,kruskal-wallis):
% r = Z/sqrt(N)
% see Feeld(2005) p.531-532, Rosenthal(1991) p.19
%
% Inputs:
% Z: z-statistic, obtained from test
% N: pooled sample size
%
% Outputs:
% r: correlation / effect size
%
% How to interpret:
% 0.1,0.3,0.5: small,medium,large
%
% Timo Flesch, 2018
r = Z/sqrt(N);

end
