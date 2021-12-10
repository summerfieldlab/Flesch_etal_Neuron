function rsa_indscal_regressBrainRDMs_parametrised_LOSO_ROI(maskName)
    %% rsa_indscal_regressBrainRDMs_parametrised_LOSO_ROI()
    %
    % regresses model rdms against brain rdms
    % in a leave-one-subject-out fashion
    % using the parametrised model.
    % training: 
    %       fit each combination of compression and rotation to n-1 subjects
    %       store best fitting parameters and use these to create predictive RDM 
    % test: use best fitting betas to generate rdm for held-out subject
    %
    % Timo Flesch, 2021,
    % Human Information Processing Lab,
    % Experimental Psychology Department
    % University of Oxford
  
    params = rsa_compute_setParams(); 
  
    grpDir = [params.dir.inDir params.dir.subDir.GRP];
  

    
    n_workers = params.hpc.numWorkers;
    gcp = parpool(n_workers)
    params.names.modelset = 'parametrised';

    % compression:
    comp_min = 0;
    comp_max = 1;
    comp_step = 0.1; % 10 percent steps to make this tractable
    comp_vect = comp_min:comp_step:comp_max;
    n_steps_comp = length(comp_vect);

    % rotation:
    phi_north = 90;
    phi_south = 0;
    phi_step = 10; % 10 degrees steps    
    rot_vect  = [-90:phi_step:0:phi_step:90];
    n_steps_rot = length(rot_vect);
  
    %% do it
    modelbetas = nan(length(params.num.goodSubjects),n_steps_comp,n_steps_rot);
    % corrs = zeros(length(params.num.goodSubjects),numMods);
    rdms  = nan(length(params.num.goodSubjects),params.num.conditions,params.num.conditions);
    for (ii = 1:length(params.num.goodSubjects))
        tic
    
        disp(['iteration ' num2str(ii) '/' num2str(length(params.num.goodSubjects))]);
        trainsubs = params.num.goodSubjects;
        trainsubs(ii) = [];

        %% gen subject matrix 
        y = [];
        for jj = 1:length(trainsubs)
            subID = trainsubs(jj);
            subStr = params.names.subjectDir(subID);
            subDir = [params.dir.inDir subStr '/' params.dir.subDir.RDM];
            cd(subDir);
            brainRDMs = load([subDir params.names.rdmSetIn maskName]);
            brainRDM = brainRDMs.subRDM.rdm;            
            % construct design matrix                               
            y_sub = vectorizeRDM(brainRDM)';
            y = cat(1,y,y_sub);
        end
        y = nanzscore(y);
        disp('constructed subject matrix')
        %% loop over over parameters 
        b = [];
        [idx_relnor,idx_relsou, idx_irrelnor,idx_irrelsou,idx_rot] = ndgrid(1:n_steps_comp,1:n_steps_comp,1:n_steps_comp,1:n_steps_comp,1:n_steps_rot);
        idx_relnor = idx_relnor(:);
        idx_relsou = idx_relsou(:);
        idx_irrelnor = idx_irrelnor(:);
        idx_irrelsou = idx_irrelsou(:);
        idx_rot = idx_rot(:);
        idces = [idx_relnor,idx_relsou,idx_irrelnor,idx_irrelsou,idx_rot];
        
        b = nan(length(idx_relnor),1);
        parfor(jj=1:size(idces,1),n_workers)        
            x = [];           
            % construct design matrix based on chosen parameter values. expand matrix and turn into z-scored regressor
            modRDM = helper_construct_modelRDM(comp_vect(idces(jj,1)),comp_vect(idces(jj,2)),comp_vect(idces(jj,3)),comp_vect(idces(jj,4)),0,rot_vect(idces(jj,5)),params.num.runs);
            x_sub = vectorizeRDM(modRDM)';
            %% concatenate training subs          
            x = repmat(x_sub,[length(trainsubs),1]);
            x = nanzscore(x);
            % get parameter estimates 
            b(jj) = regress(y,x);            
        end
        
  
        %% predict
        % find best-fitting parameters 
        [~,m] = max(b,[],'all','linear');
        % [ii_comp_hat,ii_rot_hat] = ind2sub([size(b,1),size(b,2)],m);
        % predicted rdm
        [~,rdm_pred_f] = rsa_compute_averageCvalRDMs(helper_construct_modelRDM(comp_vect(idces(m,1)),comp_vect(idces(m,2)),comp_vect(idces(m,3)),comp_vect(idces(m,4)),0,rot_vect(idces(m,5)),params.num.runs),params.num.runs,params.num.conditions);;  
               
        rdms(ii,:,:) = rdm_pred_f;
        toc
    end
  
    results = struct();
    % results.corrs = corrs;
    results.params = params;
    results.rdms = rdms;
  
    cd(grpDir);
    parsave(['results_indscal_parametrised_rdms_' maskName],results);
  
  end
  
  function parsave(str,results)
      save(str,'results');
  end
  
  function dmat = helper_construct_designMatrix(modelRDMs,ii)
    dmat = [];
    for modID = 1:length(modelRDMs)
        dmat(:,modID) = nanzscore(vectorizeRDM(squeeze(modelRDMs(modID).rdms(ii,:,:))));
    end
  end
  


  function rdm = helper_expandRDM(rdm,n_runs)

    nConds = size(rdm,2);
    rdm = repmat(rdm,[n_runs,n_runs]);
    for iiRun = 1:nConds:(nConds*n_runs)
      rdm(iiRun:iiRun+nConds-1,iiRun:iiRun+nConds-1) = NaN;
    end
  end


  function rdm = helper_construct_modelRDM(c_rel_north,c_rel_south,c_irrel_north,c_irrel_south,a1,a2,n_runs)

      % note: north=90 and south=0 are the optimal, i.e. ground-truth values
      [b,l] = meshgrid(-2:2,-2:2);
      b = b(:);
      l = l(:);
      % compress irrelevant dimension:
      respVect = [[(1-c_irrel_north).*b,(1-c_rel_north).*l];[(1-c_rel_south).*b,(1-c_irrel_south).*l]]; %l=north,b=south
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
