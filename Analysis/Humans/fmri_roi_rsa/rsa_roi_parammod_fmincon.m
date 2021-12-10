function rsa_roi_parammod_fmincon(maskName)
    %% rsa_roi_parammod_fmincon()
    %
    % fits fully parametrised model at single subject level
    % using fmincon (mse between constructed rdm and subject rdm)
    % saves the best fitting parameters (and their indices) for each subject

    % NOTE: don't forget to use -v7.3
    %
    % Timo Flesch, 2021,
    % Human Information Processing Lab,
    % Experimental Psychology Department
    % University of Oxford

    params = rsa_roi_params();

    grpDir = [params.dir.inDir params.dir.subDir.GRP];

    params.names.modelset = 'parametrised';

    N_ITERS = 500;

    % %context
    ctx_min = 0;
    ctx_max = 2;

    % compression:
    comp_min = 0.0;
    comp_max = 1.0;

    % rotation:
    phi_north = 90;
    phi_south = 0;
    phi_step = 5; % 10 degrees steps
    rot_vect = [-90:phi_step:0:phi_step:90];

    bounds_north_rel = [comp_min, comp_max];
    bounds_south_rel = [comp_min, comp_max];
    bounds_north_irrel = [comp_min, comp_max];
    bounds_south_irrel = [comp_min, comp_max];
    bounds_rot = [-90, 90];
    bounds_ctx = [ctx_min, ctx_max];
    constraints = [bounds_north_rel; bounds_south_rel; bounds_north_irrel; bounds_south_irrel; bounds_rot; bounds_ctx];

    rdms = nan(N_ITERS, length(params.num.goodSubjects), params.num.conditions, params.num.conditions);
    betas_hat = nan(N_ITERS, length(params.num.goodSubjects), 6);

    
    for (ii = 1:length(params.num.goodSubjects))
      disp(['processing subject ' num2str(ii)])
      parfor it = 1:N_ITERS
            % initvals = [.0,.0,1,1,0,2]; % optim [.0,.0,1,1,0,2]

            if contains(maskName,'EVC')
                initvals = [rand(1),rand(1),rand(1),rand(1),randsample(-90:90,1),2*rand(1)];
            else
                initvals = [.6*rand(1),.6*rand(1),.5+(1-.5)*rand(1),.5+(1-.5)*rand(1),randsample(-90:90,1),2*rand(1)];
            end
            subID = params.num.goodSubjects(ii);

            %% gen subject matrix
            subStr = params.names.subjectDir(subID);
            subDir = [params.dir.inDir subStr '/' params.dir.subDir.RDM];
            cd(subDir);
            brainRDMs = load([subDir params.names.rdmSetIn maskName]);

            brainRDM = brainRDMs.subRDM.rdm;
            [~, brainRDM] = rsa_compute_averageCvalRDMs(brainRDM, params.num.runs, params.num.conditions);

            y_sub = scale01(vectorizeRDM(brainRDM)');

            loss = @(initvals)sum((y_sub - ParamModelRDM(initvals, params.num.runs)).^2);
            [betas, minloss] = fmincon(loss, initvals, [], [], [], [], constraints(:, 1), constraints(:, 2), [], optimoptions('fmincon', 'Display', 'off'));

            betas_hat(it,ii, :) = betas;
            
            
            rdms(it,ii, :, :) = squareform(ParamModelRDM(betas, params.num.runs));      
        end        
    end
    % average over independent iterations:
    rdms = squeeze(nanmean(rdms,1));
    betas_hat = squeeze(nanmean(betas_hat,1));

    % store results
    results = struct();
    
    results.params = params;
    results.betas_hat = betas_hat;
    results.rdms = rdms;
    betas_hat(5) = abs(betas_hat(5));
    results.meanRDM = squareform(ParamModelRDM(mean(betas_hat, 1), params.num.runs));

    cd(params.dir.outDir);
    parsave(['results_fmincon_500iter_parametrised_rdms_' maskName], results);

end

function parsave(str, results)
    save(str, 'results', '-v7.3');
end

function rdm = helper_expandRDM(rdm, n_runs)

    nConds = size(rdm, 2);
    rdm = repmat(rdm, [n_runs, n_runs]);

    for iiRun = 1:nConds:(nConds * n_runs)
        rdm(iiRun:iiRun + nConds - 1, iiRun:iiRun + nConds - 1) = NaN;
    end

end

function x_sub = ParamModelRDM(theta, n_runs)
    c_rel_north = theta(1);
    c_rel_south = theta(2);
    c_irrel_north = theta(3);
    c_irrel_south = theta(4);
    a1 = 0;
    a2 = theta(5);
    ctx = theta(6);
    % note: north=90 and south=0 are the optimal, i.e. ground-truth values
    [b, l] = meshgrid(-2:2, -2:2);
    b = b(:);
    l = l(:);
    % compress irrelevant dimension:
    respVect = [[(1 - c_irrel_north) .* b, (1 - c_rel_north) .* l]; [(1 - c_rel_south) .* b, (1 - c_irrel_south) .* l]]; %l=north,b=south
    % rotate vector
    respVect(1:25, :) = respVect(1:25, :) * [cos(deg2rad(a1)), -sin(deg2rad(a1)); sin(deg2rad(a1)), cos(deg2rad(a1))];
    respVect(26:end, :) = respVect(26:end, :) * [cos(deg2rad(a2)), -sin(deg2rad(a2)); sin(deg2rad(a2)), cos(deg2rad(a2))];
    respVect = [respVect [zeros(length(b), 1); ctx + zeros(length(b), 1)]];
    rdm = squareform(pdist(respVect));
    %   rdm = helper_expandRDM(rdm,n_runs);
    x_sub = scale01(vectorizeRDM(rdm)');

end

function b = helper_fitRDMs(brainRDM, comp_vect, vect_south, n_steps_comp, n_steps_rot, n_runs)
    b = [];

    for jj = 1:n_steps_comp

        for kk = 1:n_steps_rot
            modRDM = helper_construct_modelRDM(comp_vect(jj), 0, 0, vect_south(kk), n_runs);
            X = nanzscore(vectorizeRDM(modRDM))';
            Y = nanzscore(vectorizeRDM(brainRDM))';
            b(jj, kk) = regress(Y, X);
        end

    end

end
