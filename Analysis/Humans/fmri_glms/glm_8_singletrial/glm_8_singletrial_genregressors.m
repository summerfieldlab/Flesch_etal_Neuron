function glm_8_singletrial_genregressors(allData)
    %% FMRI_UNIVAR_GENREGRESSORFILES(allData)
    %
    % generates files with multiple conditions for
    % each run and each subject to speed up
    % constructions of SPM design matrices
    %
    % Timo Flesch, 2018,
    % Human Information Processing Lab,
    % Experimental Psychology Department
    % University of Oxford
    params = glm_8_singletrial_params();

    numSubs = length(allData.order);
    numRuns = length(unique(allData.expt_block(1, :)));

    for subID = 1:numSubs

        for runID = 1:numRuns
            names = {};
            onsets = {};
            durations = {};

            %% stim onset
            
            t_stim = [allData.time_onset_stim(subID, allData.expt_block(subID, :) == runID) - allData.time_trigRun(subID, runID)];
            bidcs = allData.expt_branchIDX(subID, allData.expt_block(subID, :) == runID);
            lidcs = allData.expt_leafIDX(subID, allData.expt_block(subID, :) == runID);
            cidcs = allData.expt_contextIDX(subID, allData.expt_block(subID,:) == runID); 
            for t = 1:length(t_stim)                
                names{t} = ['run' num2str(runID) '_b' num2str(bidcs(t)) '_l' num2str(lidcs(t)) '_c' num2str(cidcs(t))];
                onsets{t} = t_stim(t);
                durations{t} = 0;                
            end            
            save([params.dir.conditionDir 'conditions_' params.glmName '_sub' num2str(subID) '_run' num2str(runID)], 'names', 'onsets', 'durations');

        end

    end

end
