function [ep,pep,emf] = rsa_roi_runBMS(maskname)
    %% RSA_ROI_RUNBMS(maskname)
    %
    % performs bayesian model selection 
    % for a group of regression models
    % within a cross-validated ROI
    %
    % Timo Flesch, 2021

    % load log-model-evidence for each model
    load(['glmfit_grp_modelbranchortho_betas_cval_searchlight_' maskname '.mat']);
    lmes = [results.lmes];
    load(['glmfit_grp_modelonlybranch_betas_cval_searchlight_' maskname '.mat']);
    lmes = [lmes,results.lmes];
    load(['glmfit_grp_modelonlyortho_betas_cval_searchlight_' maskname '.mat']);
    lmes = [lmes,results.lmes];
    lmes = lmes'; % models-by-subjects
    % perform bms
    [posterior, out] = VBA_groupBMC(lmes);
    % return (protected) exceedance probabilities and estimated model frequencies 
    pep = (1-out.bor)*out.ep + out.bor/length(out.ep);
    ep = out.ep;
    emf = posterior.r;
    close all;
    save(['bmsresults_branchortho_branch_ortho_' maskname '.mat'],'ep','pep','emf');

end