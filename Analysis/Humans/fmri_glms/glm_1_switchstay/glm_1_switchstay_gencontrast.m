function contrasts = glm_1_switchstay_gencontrast()
    %% FMR_UNIVAR_GENCONTRASTS()
    %
    % specifies contrasts
    %
    % Timo Flesch, 2018
    % Human Information Processing Lab
    % Experimental Psychology Department
    % University of Oxford

    params = glm_1_switchstay_params();
    contrasts = struct();

    % give the contrasts sensible names
    contrasts.T.labels = {'Stay > Switch', 'Switch > Stay'};
    % define the contrast vectors ( [repmat([CONDITIONS, MOTIONREGS],1,numRuns), RUNIDS] )
    contrasts.T.vectors(1, :) = helper_genContrastVector(2, 1, 2, params);
    contrasts.T.vectors(2, :) = helper_genContrastVector(2, 2, 1, params);

end
