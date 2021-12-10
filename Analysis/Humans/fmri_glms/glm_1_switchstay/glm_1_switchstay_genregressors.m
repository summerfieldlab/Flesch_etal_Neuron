function glm_1_switchstay_genregressors(allData)
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
    params = glm_1_switchstay_params();

    numSubs = length(allData.order);
    numRuns = length(unique(allData.expt_block(1, :)));

    for subID = 1:numSubs

        for runID = 1:numRuns
            names = {};
            onsets = {};
            durations = {};

            % %% fix onset
            t_fix = [allData.time_onset_fix(subID, allData.expt_block(subID, :) == runID) - allData.time_trigRun(subID, runID)];
            cueTime = mean(allData.time_onset_stim(:) - allData.time_onset_fix(:));
            resp_stim = allData.resp_reactiontime(subID, allData.expt_block(subID, :) == runID) + cueTime;
            resp_stim(isnan(resp_stim)) = 1.5; % set duration for missed trials to length of stimulus interval
            % names{1}     = 'stim_onset';
            % onsets{1}    =  t_fix;
            % durations{1} =  0;

            %% context stay
            ctxVect = squeeze(allData.expt_contextIDX(subID, allData.expt_block(subID, :) == runID));
            stayVect = ctxVect == circshift(ctxVect, [1, 1]);
            tStay = t_fix;
            respStay = resp_stim;
            % stayVect(1)  =        [];
            % tStay(1)     =        [];
            % respStay(1)  =        [];
            tStay = tStay(stayVect); % discard first entry as it doesn't have a predecessor
            respStay = respStay(stayVect); % ditto
            names{1} = 'ctxStay';
            onsets{1} = tStay;
            durations{1} = respStay;

            %% context switch
            switchVect = ctxVect ~= circshift(ctxVect, [1, 1]);
            tSwitch = t_fix;
            respSwitch = resp_stim;
            % switchVect(1)  =        [];
            % tSwitch(1)     =        [];
            % respSwitch(1)  =        [];
            tSwitch = tSwitch(switchVect); % discard first entry as it doesn't have a predecessor
            respSwitch = respSwitch(switchVect); % ditto
            names{2} = 'ctxSwitch';
            onsets{2} = tSwitch;
            durations{2} = respSwitch;

            save([params.dir.conditionDir 'conditions_' params.glmName '_sub' num2str(subID) '_run' num2str(runID)], 'names', 'onsets', 'durations');

        end

    end

end
