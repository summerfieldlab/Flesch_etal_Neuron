function v = helper_genWeightedContrastVector(n,iiPos,iiNeg,wPos,wNeg,params)
  %% HELPER_GENCONTRASTVECTOR(N,IIPOS,IINEG)
  %
  % n:     number of conditions
  % iiPos: indices of positive weights
  % iiNeg: indices of negative weights
  %
  % for example, if 10 conditions and looking for main effect of third condition,
  % type:
  % v = helper_genContrastVector(10,[3],[]);
  % Timo Flesch, 2018

  allRunsVect = [];
  for runID = 1:length(wPos)
    conditions = zeros(1,n);
    conditions(iiPos) =  1*wPos(runID);
    conditions(iiNeg) = -1*wNeg(runID);
    allRunsVect = [allRunsVect conditions zeros(1,params.num.motionregs)];
  end
  v = [allRunsVect  zeros(1,params.num.runs)]; % conditions + motion regressors and dummy vars for runs
end
