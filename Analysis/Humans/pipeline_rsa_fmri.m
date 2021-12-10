function pipeline_rsa_fmri()
    %
    % pipeline for RSA results reported in paper

    roilist = {'leftEVC_all7mods_mod5_', 'rightDLPFC_all7mods_mod3_', 'rightMCC_all7mods_mod3_', 'rightPAR_all7mods_mod3_'};

    
    % run_searchlight_rsa();
    
    % run_loso_sl_rsa();
    
    % run_roi_rsa();

    % run_roi_bms_rsa();
    % run_roi_svd();
    
    % run_roi_svd_rsa();

    run_roi_parammod_rsa();


    function run_searchlight_rsa()
        % NOTE: for statistical inference, we actually used the SNPM toolbox
        % instead of the parametric approach shown below.
        % 1. regress brain rdms against model rdms within sl spheres:
        rsa_searchlight_regressmodels();
        % 2. create .nii images from estimated beta maps:
        rsa_searchlight_genbetaimages();
        % 3. perform 2nd level analysis on beta maps and create t-images:
        rsa_searchlight_sigtest();
        rsa_searchlight_gensigimages();

    end

    function run_loso_sl_rsa()
        %% Searchlight RSA: LOSO:
        % leave-one-subject-out rsa is carried out to estimate cross-validated
        % functional ROIs (the ones used for ROI analyses shown below)
        rsa_searchlight_loso_sigtest();
        rsa_searchlight_loso_gensigimages();
    end

    function run_roi_rsa()

        params = rsa_roi_params();

        if params.hpc.parallelise
            gcp = parpool(params.hpc.numWorkers)
        end

        % perform rsa within functional ROIs:
        for maskID = 1:length(roilist)
            mask = roilist{maskID};
            disp(['ROI RSA with mask ' mask]);

            rsa_roi_compute(mask);
            rsa_roi_regressmodels(mask);
            rsa_roi_sigtest(mask);

            disp(['...done']);
        end

        if params.hpc.parallelise
            delete(gcp)
        end

    end

    function run_roi_bms_rsa()

        params = rsa_roi_params();

       
        % perform rsa within functional ROIs:
        for maskID = 1:length(roilist)
            mask = roilist{maskID};
            disp(['ROI GLMFIT RSA with mask ' mask]);

            % full model:
            rsa_roi_glmfit(mask);
            
            % without branchiness:
            rsa_roi_glmfit(mask,[1,2,3,4,6,7],'nobranchiness');

            % without orthogonal
            rsa_roi_glmfit(mask,[1,2,4,5,6,7],'noorthogonal');

            % only branch n ortho
            rsa_roi_glmfit(mask,[3,5],'branchortho');

            % only branch
            rsa_roi_glmfit(mask,[5],'onlybranch');

            % only ortho
            rsa_roi_glmfit(mask,[3],'onlyortho');


            % perform bayesian model selection:
            rsa_roi_runBMS(mask);


            disp(['...done']);
        end

       
    end

    function betas = run_roi_svd_rsa()

        params = rsa_roi_params();

        if params.hpc.parallelise
            gcp = parpool(params.hpc.numWorkers)
        end

        % perform rsa within functional ROIs:
        for maskID = 1:length(roilist)
            mask = roilist{maskID};
            disp(['ROI RSA with mask ' mask]);

            parfor (d = 1:50, 6)
                rsa_roi_svd_compute(mask, d);
                rsa_roi_svd_regressmodels(mask, d);
                rsa_roi_svd_sigtest(mask, d);
            end

            disp(['...done']);
        end

        if params.hpc.parallelise
            delete(gcp)
        end     

    end

    function run_roi_svd()
         % perform rsa within functional ROIs:
         for maskID = 1:length(roilist)
            mask = roilist{maskID};
            disp(['ROI SVD with mask ' mask]);
            rsa_roi_svd_singularvals(mask); 
            
            disp(['...done']);
        end

    end

    function run_roi_parammod_rsa()

        params = rsa_roi_params();

        % perform rsa within functional ROIs:
        for maskID = 1:length(roilist)
            mask = roilist{maskID};
            disp(['ROI Parammod with mask ' mask]);

            rsa_roi_parammod_fmincon(mask);
  

            disp(['...done']);
        end
    end

end
