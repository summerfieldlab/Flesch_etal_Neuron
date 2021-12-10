function rsa_corrs_regressBrainRDMs_ROI_CompressAngle(maskName)
  %% rsa_corrs_regressBrainRDMs_ROI_CompressionBoth(maskName,modelRDMs)
  %
  % fits parametric model with rotation angle and compression along irrelevant
  % as free parameters
  %
  % Yields a surface of correlation coefficients (taskA x taskB)
  % where the peak should in theory lie at 0,1
  %
  % Timo Flesch, 2019,
  % Human Information Processing Lab,
  % Experimental Psychology Department
  % University of Oxford

    params = rsa_compute_setParams();
    n_workers = params.hpc.numWorkers;
    params.names.modelset = 'compressangle';

    grpDir = [params.dir.inDir params.dir.subDir.GRP];

    % compression:
    comp_min = 0;
    comp_max = 1;
    comp_step = 0.01; % 1 percent steps
    comp_vect = comp_min:comp_step:comp_max;
    n_steps_comp = length(comp_vect);

    % rotation:
    phi_north = 90;
    phi_south = 0;
    phi_step = 1;
    % vect_north = [0:phi_step:180];
    vect_south  = [-90:phi_step:0:phi_step:90];
    n_steps_rot = length(vect_south);


    % gcp = parpool(n_workers)
    %% do it

    modelbetas = nan(length(params.num.goodSubjects),n_steps_comp,n_steps_rot);
    parfor (ii = 1:length(params.num.goodSubjects),n_workers)
      disp(['processing subject ' num2str(ii)]);
      subID = params.num.goodSubjects(ii);
      subStr = params.names.subjectDir(subID);
      subDir = [params.dir.inDir subStr '/' params.dir.subDir.RDM];
      cd(subDir);

      brainRDMs = load([subDir params.names.rdmSetIn maskName]);
      brainRDM = brainRDMs.subRDM.rdm;
      modelbetas(ii,:,:) = helper_fitRDMs(brainRDM,comp_vect,vect_south,n_steps_comp,n_steps_rot,params.num.runs);
      disp(['...finished subject ' num2str(ii)]);
    end
    % delete(gcp);

    results = modelbetas;
    cd(grpDir);
    save(['results_compressangle_' maskName],'results');
  end


  function rdm = helper_expandRDM(rdm,n_runs)

    nConds = size(rdm,2);
    rdm = repmat(rdm,[n_runs,n_runs]);
    for iiRun = 1:nConds:(nConds*n_runs)
      rdm(iiRun:iiRun+nConds-1,iiRun:iiRun+nConds-1) = NaN;
    end
  end


  function rdm = helper_construct_modelRDM(c_irrel,c_rel,a1,a2,n_runs)

      % note: north=90 and south=0 are the optimal, i.e. ground-truth values
      [b,l] = meshgrid(-2:2,-2:2);
      b = b(:);
      l = l(:);
      % compress irrelevant dimension:
      respVect = [[(1-c_irrel).*b,(1-c_rel).*l];[(1-c_rel).*b,(1-c_irrel).*l]]; %l=north,b=south
      % rotate vector
      respVect(1:25,:) = respVect(1:25,:)*[cos(deg2rad(a1)),-sin(deg2rad(a1));sin(deg2rad(a1)),cos(deg2rad(a1))];
      respVect(26:end,:) = respVect(26:end,:)*[cos(deg2rad(a2)),-sin(deg2rad(a2));sin(deg2rad(a2)),cos(deg2rad(a2))];
      rdm = squareform(pdist(respVect));
      rdm = helper_expandRDM(rdm,n_runs);

  end

  function b = helper_fitRDMs(brainRDM,comp_vect,vect_south,n_steps_comp, n_steps_rot,n_runs)
      b = [];
      for jj = 1:n_steps_comp
          for kk = 1:n_steps_rot
              modRDM = helper_construct_modelRDM(comp_vect(jj),0,0,vect_south(kk),n_runs);
              X = nanzscore(vectorizeRDM(modRDM))';
              Y = nanzscore(vectorizeRDM(brainRDM))';
              b(jj,kk) = regress(Y,X);
          end
      end
  end
