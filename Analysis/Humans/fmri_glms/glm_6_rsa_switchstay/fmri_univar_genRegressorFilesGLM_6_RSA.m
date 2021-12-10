function bad_trials = fmri_univar_genRegressorFilesGLM_6_RSA(allData)
  %
  % design matrix for rsa on switch trials
  %
  % Timo Flesch, 2018,
  % Human Information Processing Lab,
  % Experimental Psychology Department
  % University of Oxford
  params = fmri_univar_setParamsGLM_6_RSA();

  numSubs = length(allData.order);
  numRuns = length(unique(allData.expt_block(1,:)));
  bad_trials = zeros(numSubs,600);

  for subID = 1:numSubs
    iTrial = 1;
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
        dataCtx = allData.expt_contextIDX(subID,allData.expt_block(subID,:)==runID);
        dataBranch = allData.expt_branchIDX(subID,allData.expt_block(subID,:)==runID);
        dataLeaf = allData.expt_leafIDX(subID,allData.expt_block(subID,:)==runID);
        ctxVect = squeeze(allData.expt_contextIDX(subID,allData.expt_block(subID,:)==runID));
        stayVect = ctxVect == circshift(ctxVect,[1,1]);
        switchVect     = ctxVect ~= circshift(ctxVect,[1,1]);

        % all conditions on stay trials
        iCondition = 1;
        for iCtx = 1:2
          for iBranch = 1:5
            for iLeaf = 1:5
              % STAY trials with context, branchiness and leafiness level (might be zero..)
              iTrials = find(dataCtx==iCtx & dataBranch==iBranch & dataLeaf==iLeaf & stayVect==1);
              names{iCondition} = ['C' num2str(iCtx) 'B' num2str(iBranch) 'L' num2str(iLeaf) 'STAY'];
              % if no data, set onset to last image
              if isempty(iTrials)
                onsets{iCondition}    = t_stim(end);
                bad_trials(subID,iTrial) = 1;
              else
                onsets{iCondition}    = t_stim(iTrials);
              end
              durations{iCondition} = 0; %NOTE now stick functions instead of these boxcars: resp_stim(iTrials);
              iCondition = iCondition + 1;
              iTrial = iTrial + 1;
            end
          end
        end

        %% all conditions on switch trials
        for iCtx = 1:2
          for iBranch = 1:5
            for iLeaf = 1:5
              % STAY trials with context, branchiness and leafiness level (might be zero..)
              iTrials = find(dataCtx==iCtx & dataBranch==iBranch & dataLeaf==iLeaf & switchVect==1);
              names{iCondition} = ['C' num2str(iCtx) 'B' num2str(iBranch) 'L' num2str(iLeaf) 'SWITCH'];
              % if no data, set onset to last image
              if isempty(iTrials)
                onsets{iCondition}    = t_stim(end);
                bad_trials(subID,iTrial) = 1;
              else
                onsets{iCondition}    = t_stim(iTrials);
              end
              durations{iCondition} = 0; %NOTE now stick functions instead of these boxcars: resp_stim(iTrials);
              iCondition = iCondition + 1;
              iTrial = iTrial + 1;
            end
          end
        end

        save([params.dir.conditionDir 'conditions_' params.glmName '_sub' num2str(subID) '_run' num2str(runID)],'names','onsets','durations');

    end
  end



end
