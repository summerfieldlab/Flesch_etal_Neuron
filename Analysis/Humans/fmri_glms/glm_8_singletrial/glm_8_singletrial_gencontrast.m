function contrasts = glm_8_singletrial_gencontrast(subID, allData)
    %% FMR_UNIVAR_GENCONTRASTS()
    %
    % specifies contrasts
    %
    % Timo Flesch, 2018
    % Human Information Processing Lab
    % Experimental Psychology Department
    % University of Oxford

    params = glm_8_singletrial_params();
    contrasts = struct();

    % give the contrasts sensible names
    n_trials = 600;
    contrasts.T.labels = {};
    idx = 1;
    idx_col = 1;
    contrasts.T.vectors = zeros(n_trials,params.num.runs*(100+params.num.motionregs)+params.num.runs);
    for runID = 1:6
        t_stim = [allData.time_onset_stim(subID, allData.expt_block(subID, :) == runID) - allData.time_trigRun(subID, runID)];
        bidcs = allData.expt_branchIDX(subID, allData.expt_block(subID, :) == runID);
        lidcs = allData.expt_leafIDX(subID, allData.expt_block(subID, :) == runID);
        cidcs = allData.expt_contextIDX(subID, allData.expt_block(subID,:) == runID); 
        for t = 1:length(t_stim)               
    
            contrasts.T.labels{idx} = ['run' num2str(runID) '_b' num2str(bidcs(t)) '_l' num2str(lidcs(t)) '_c' num2str(cidcs(t))];
            contrasts.T.vectors(idx, idx_col) = 1;
    
            idx = idx + 1;
            idx_col = idx_col + 1;
        end
        idx_col = idx_col + params.num.motionregs;
    end
    idx_col = idx_col + params.num.runs;    
    debug

end
