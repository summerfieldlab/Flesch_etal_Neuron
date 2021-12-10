
function bic = compute_BIC(logL,k,n)
	% computes BIC
	% logL = log-likelihood
	% k = number of parameters
	% n = number of observations
	% 
	% Timo Flesch, 2018

	 bic = -2*logL + k.*log(n);
end