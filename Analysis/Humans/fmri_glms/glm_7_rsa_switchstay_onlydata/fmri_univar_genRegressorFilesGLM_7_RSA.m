function trial_ids = fmri_univar_genRegressorFilesGLM_7_RSA(allData)
  %
  % design matrix for rsa on switch trials
  %
  % Timo Flesch, 2018,
  % Human Information Processing Lab,
  % Experimental Psychology Department
  % University of Oxford
  params = fmri_univar_setParamsGLM_7_RSA();

  numSubs = length(allData.order);
  numRuns = length(unique(allData.expt_block(1,:)));
  trial_ids = {};

  for subID = 1:numSubs
    iTrial = 1;
    has_data = []; zeros(numRuns,2,2,5,5);
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
              % select condition only if it has trials
              if ~isempty(iTrials)                
                names{iCondition} = ['C' num2str(iCtx) 'B' num2str(iBranch) 'L' num2str(iLeaf) 'STAY'];
                onsets{iCondition}    = t_stim(iTrials);
                % has_data(runID,1,iCtx,iBranch,iLeaf) = 1;
                has_data(iTrial,1) = 1;
                has_data(iTrial,2) = runID;
                has_data(iTrial,3) = iCtx;
                has_data(iTrial,4) = iBranch;
                has_data(iTrial,5) = iLeaf;                
                durations{iCondition} = 0; %NOTE now stick functions instead of these boxcars: resp_stim(iTrials);
                iCondition = iCondition + 1;
                iTrial = iTrial + 1;                
              end
            end
          end
        end

        %% all conditions on switch trials
        for iCtx = 1:2
          for iBranch = 1:5
            for iLeaf = 1:5
              % STAY trials with context, branchiness and leafiness level (might be zero..)
              iTrials = find(dataCtx==iCtx & dataBranch==iBranch & dataLeaf==iLeaf & switchVect==1);
              % only select conditions with data points
              if ~isempty(iTrials)                
                names{iCondition} = ['C' num2str(iCtx) 'B' num2str(iBranch) 'L' num2str(iLeaf) 'SWITCH'];
                onsets{iCondition}    = t_stim(iTrials);
                % has_data(runID,1,iCtx,iBranch,iLeaf) = 1;
                has_data(iTrial,1) = 2;
                has_data(iTrial,2) = runID;
                has_data(iTrial,3) = iCtx;
                has_data(iTrial,4) = iBranch;
                has_data(iTrial,5) = iLeaf;                
                durations{iCondition} = 0; %NOTE now stick functions instead of these boxcars: resp_stim(iTrials);
                iCondition = iCondition + 1;
                iTrial = iTrial + 1;                
              end
            end
          end
        end
        save([params.dir.conditionDir 'conditions_' params.glmName '_sub' num2str(subID) '_run' num2str(runID)],'names','onsets','durations');

    end
    trial_ids{subID} = has_data;
  end



end
