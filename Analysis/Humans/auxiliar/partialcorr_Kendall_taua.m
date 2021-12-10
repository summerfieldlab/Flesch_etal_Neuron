function taua_bc_a = partialcorr_Kendall_taua(RDM_ref,RDM_cand1,RDM_cand2)
%
% computes partial rank correlation between candidate  and reference RDM
% as correlation between cand1 and ref, keeping cand2 constant
%
% (c) Timo Flesch, 2016
% Summerfield Lab, Experimental Psychology Department, University of Oxford
%
% reference: Kendall (1942)



if ~isvector(RDM_ref)
	RDM_ref = vectorizeRDM(RDM_ref);
end
if ~isvector(RDM_cand1)
	RDM_cand1 = vectorizeRDM(RDM_cand1);
end
if ~isvector(RDM_cand2)
	RDM_cand2 = vectorizeRDM(RDM_cand2);
end

taua_bc = rankCorr_Kendall_taua(RDM_cand1,RDM_ref);
taua_ab = rankCorr_Kendall_taua(RDM_cand2,RDM_cand1);
taua_ac = rankCorr_Kendall_taua(RDM_cand2,RDM_ref);
taua_bc_a = (taua_bc-taua_ab*taua_ac)/(sqrt(1-taua_ab^2)*sqrt(1-taua_ac^2));

end
