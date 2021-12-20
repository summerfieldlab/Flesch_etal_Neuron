function glm_3b_signeddist_choice_genregressors(allData, rsData)
    %% glm_3b_signeddist_choice_genregressors(allData)
    %
    % generates files with multiple conditions for
    % each run and each subject to speed up
    % constructions of SPM design matrices
    %
    % Timo Flesch, 2018,
    % Human Information Processing Lab,
    % Experimental Psychology Department
    % University of Oxford
    params = glm_3b_signeddist_choice_params();

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

            %% reward for planting a tree, parametric, relevant dimension
            rewards = rsData(subID).data(rsData(subID).data(:, 1) == runID, 7)';
            pmod(1).name{1} = 'signedDistance_relDim';
            pmod(1).param{1} = rewards;
            pmod(1).poly{1} = 1;

            %% reward for planting a tree, parametric, irrelevant dimension
            rewards = rsData(subID).data(rsData(subID).data(:, 1) == runID, 14)';
            pmod(1).name{2} = 'signedDistance_irrelDim';
            pmod(1).param{2} = rewards;
            pmod(1).poly{2} = 1;

            orth{1} = 0;

            % choice (accept/reject)
            trials = t_stim+resp_stim;
            accept = allData.resp_category(subID, allData.expt_block(subID, :) == runID);
            
            names{2} = 'accept';
            onsets{2} = trials(accept==1);
            durations{2} = 0;


            % choice x rel dim            
            rewards = rsData(subID).data(rsData(subID).data(:, 1) == runID, 7)';
            pmod(2).name{1} = 'choice_x_relDim';
            pmod(2).param{1} = rewards(accept==1);
            pmod(2).poly{1} = 1;

            % choice x irrel dim
            rewards = rsData(subID).data(rsData(subID).data(:, 1) == runID, 14)';
            pmod(2).name{2} = 'choice_x_irrelDim';
            pmod(2).param{2} = rewards(accept==1);
            pmod(2).poly{2} = 1;
            
            orth{2} = 0;

            save([params.dir.conditionDir 'conditions_' params.glmName '_sub' num2str(subID) '_run' num2str(runID)], 'names', 'onsets', 'durations', 'pmod', 'orth');

        end

    end

end
