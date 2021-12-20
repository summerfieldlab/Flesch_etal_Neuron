function contrasts = glm_3b_signeddist_choice_gencontrast()
%% glm_3_signeddist_gencontrast()
%
% specifies contrasts
%
% Timo Flesch, 2018
% Human Information Processing Lab
% Experimental Psychology Department
% University of Oxford

  if ~exist('monitor','var')
    monitor = 0;
  end

  params = glm_3b_signeddist_choice_params();
  contrasts = struct();

  % give the contrasts sensible names
  contrasts.T.labels = {'signeddist rel','signeddist irrel', 'choice', 'choice x signeddist rel', 'choice x signeddist irrel' };
  % define the contrast vectors ( [repmat([CONDITIONS, MOTIONREGS],1,numRuns), RUNIDS] )
  contrasts.T.vectors(1,:) = helper_genContrastVector(6,2,[],params);
  contrasts.T.vectors(2,:) = helper_genContrastVector(6,3,[],params);
  contrasts.T.vectors(3,:) = helper_genContrastVector(6,4,[],params);
  contrasts.T.vectors(4,:) = helper_genContrastVector(6,5,[],params);
  contrasts.T.vectors(5,:) = helper_genContrastVector(6,6,[],params);

end
