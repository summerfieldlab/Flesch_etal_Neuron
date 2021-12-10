function glm_2_absdist_genregressors(allData)
    %% glm_2_absdist_genregressors(allData)
    %
    % generates files with multiple conditions for
    % each run and each subject to speed up
    % constructions of SPM design matrices
    %
    % Timo Flesch, 2018,
    % Human Information Processing Lab,
    % Experimental Psychology Department
    % University of Oxford
    params = glm_2_absdist_params();

    numSubs = length(allData.order);
    numRuns = length(unique(allData.expt_block(1, :)));

    for subID = 1:numSubs

        for runID = 1:numRuns
            names = {};
            onsets = {};
            durations = {};

            % %% stim onset
            t_stim = [allData.time_onset_stim(subID, allData.expt_block(subID, :) == runID) - allData.time_trigRun(subID, runID)];
            resp_stim = allData.resp_reactiontime(subID, allData.expt_block(subID, :) == runID);
            resp_stim(isnan(resp_stim)) = 1.5; % set duration for missed trials to length of stimulus interval

            names{1} = 'stimulus';
            onsets{1} = t_stim;
            durations{1} = resp_stim;
            %% tree: relevant dimension: parametric modulator
            idces_hiDist = find((allData.expt_contextIDX(subID, allData.expt_block(subID, :) == runID) == 1 & (allData.expt_leafIDX(subID, allData.expt_block(subID, :) == runID) == 1 | allData.expt_leafIDX(subID, allData.expt_block(subID, :) == runID) == 5)) | (allData.expt_contextIDX(subID, allData.expt_block(subID, :) == runID) == 2 & (allData.expt_branchIDX(subID, allData.expt_block(subID, :) == runID) == 1 | allData.expt_branchIDX(subID, allData.expt_block(subID, :) == runID) == 5)));
            idces_loDist = find((allData.expt_contextIDX(subID, allData.expt_block(subID, :) == runID) == 1 & (allData.expt_leafIDX(subID, allData.expt_block(subID, :) == runID) == 2 | allData.expt_leafIDX(subID, allData.expt_block(subID, :) == runID) == 4)) | (allData.expt_contextIDX(subID, allData.expt_block(subID, :) == runID) == 2 & (allData.expt_branchIDX(subID, allData.expt_block(subID, :) == runID) == 2 | allData.expt_branchIDX(subID, allData.expt_block(subID, :) == runID) == 4)));
            idces_bound = find((allData.expt_contextIDX(subID, allData.expt_block(subID, :) == runID) == 1 & (allData.expt_leafIDX(subID, allData.expt_block(subID, :) == runID) == 3)) | (allData.expt_contextIDX(subID, allData.expt_block(subID, :) == runID) == 2 & (allData.expt_branchIDX(subID, allData.expt_block(subID, :) == runID) == 3)));
            distToBound = ones(length(t_stim), 1) .* 1000;
            distToBound(idces_hiDist) = 2;
            distToBound(idces_loDist) = 1;
            distToBound(idces_bound) = 0;

            pmod(1).name{1} = 'paramRelDistToBound';
            pmod(1).param{1} = zscore(distToBound);
            pmod(1).poly{1} = 1;

            %% trees: irrelevant dimension: parametric modulator
            idces_hiDist = find((allData.expt_contextIDX(subID, allData.expt_block(subID, :) == runID) == 1 & (allData.expt_branchIDX(subID, allData.expt_block(subID, :) == runID) == 1 | allData.expt_branchIDX(subID, allData.expt_block(subID, :) == runID) == 5)) | (allData.expt_contextIDX(subID, allData.expt_block(subID, :) == runID) == 2 & (allData.expt_leafIDX(subID, allData.expt_block(subID, :) == runID) == 1 | allData.expt_leafIDX(subID, allData.expt_block(subID, :) == runID) == 5)));
            idces_loDist = find((allData.expt_contextIDX(subID, allData.expt_block(subID, :) == runID) == 1 & (allData.expt_branchIDX(subID, allData.expt_block(subID, :) == runID) == 2 | allData.expt_branchIDX(subID, allData.expt_block(subID, :) == runID) == 4)) | (allData.expt_contextIDX(subID, allData.expt_block(subID, :) == runID) == 2 & (allData.expt_leafIDX(subID, allData.expt_block(subID, :) == runID) == 2 | allData.expt_leafIDX(subID, allData.expt_block(subID, :) == runID) == 4)));
            idces_bound = find((allData.expt_contextIDX(subID, allData.expt_block(subID, :) == runID) == 1 & (allData.expt_branchIDX(subID, allData.expt_block(subID, :) == runID) == 3)) | (allData.expt_contextIDX(subID, allData.expt_block(subID, :) == runID) == 2 & (allData.expt_leafIDX(subID, allData.expt_block(subID, :) == runID) == 3)));
            distToBound = ones(length(t_stim), 1) .* 1000;
            distToBound(idces_hiDist) = 2;
            distToBound(idces_loDist) = 1;
            distToBound(idces_bound) = 0;
            pmod(1).name{2} = 'paramIrrelDistToBound';
            pmod(1).param{2} = zscore(distToBound);
            pmod(1).poly{2} = 1;

            orth{1} = 0;

            save([params.dir.conditionDir 'conditions_' params.glmName '_sub' num2str(subID) '_run' num2str(runID)], 'names', 'onsets', 'durations', 'pmod', 'orth');

        end

    end

end
