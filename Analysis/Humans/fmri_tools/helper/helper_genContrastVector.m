function v = helper_genContrastVector(n,iiPos,iiNeg,params)
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

  conditions = zeros(1,n);
  conditions(iiPos) =  1;
  conditions(iiNeg) = -1;

  v = [repmat([conditions zeros(1,params.num.motionregs)],1,params.num.runs)  zeros(1,params.num.runs)];

end
