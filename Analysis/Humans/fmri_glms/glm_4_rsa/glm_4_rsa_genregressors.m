function glm_4_rsa_genregressors(allData)
  %% glm_4_rsa_genregressors(allData)
  %
  % generates files with multiple conditions for
  % each run and each subject to speed up
  % constructions of SPM design matrices
  %
  % Timo Flesch, 2018,
  % Human Information Processing Lab,
  % Experimental Psychology Department
  % University of Oxford
  params = glm_4_rsa_params();

  numSubs = length(allData.order);
  numRuns = length(unique(allData.expt_block(1,:)));
  for subID = 1:numSubs
    for runID = 1:numRuns
        names     = {};
        onsets    = {};
        durations = {};

        % %% stim onset
        t_stim = [allData.time_onset_stim(subID,allData.expt_block(subID,:)==runID) - allData.time_trigRun(subID,runID)];
        resp_stim = allData.resp_reactiontime(subID,allData.expt_block(subID,:)==runID);
        resp_stim(isnan(resp_stim)) = 1.5; % set duration for missed trials to length of stimulus interval
        % names{1}     = 'stim_onset';
        % onsets{1}    =  t_stim;
        % durations{1} =  0;

        %% (one column per context-x-branchiness-x-leafinesss + 6 nuisance)
        iCondition = 1;
        for iCtx = 1:2
          for iBranch = 1:5
            for iLeaf = 1:5
              dataCtx = allData.expt_contextIDX(subID,allData.expt_block(subID,:)==runID);
              dataBranch = allData.expt_branchIDX(subID,allData.expt_block(subID,:)==runID);
              dataLeaf = allData.expt_leafIDX(subID,allData.expt_block(subID,:)==runID);
              iTrials = find(dataCtx==iCtx & dataBranch==iBranch & dataLeaf==iLeaf);
              names{iCondition} = ['C' num2str(iCtx) '_B' num2str(iBranch) '_L' num2str(iLeaf)];
              onsets{iCondition}    = t_stim(iTrials);
              durations{iCondition} = 0; %NOTE now stick functions instead of these boxcars: resp_stim(iTrials);
              iCondition = iCondition + 1;
            end
          end
        end


        save([params.dir.conditionDir 'conditions_' params.glmName '_sub' num2str(subID) '_run' num2str(runID)],'names','onsets','durations');

    end
  end



end
