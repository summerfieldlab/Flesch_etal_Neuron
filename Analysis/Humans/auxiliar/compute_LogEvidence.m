function lme = compute_LogEvidence(bic)
	% computes approximation of log model evidence
	%
	% Timo Flesch, 2018

	lme = -.5.*bic;
end