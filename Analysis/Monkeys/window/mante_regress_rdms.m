function betas = mante_regress_rdms(params,results,withinTasks)
    %% mante_regress_rdms(results)
    % 
    % regresses brain rdms against set of candidate rdms 
    % for each monkey and returns param estimates 
    %
    % if desired, plot beta estimates 
    % 
    % Timo Flesch

    if ~exist('withinTasks')
        withinTasks = false;
    end 
    modelRDMs = mante_gen_modelrdms(params);
    if withinTasks==true
        tmp = struct();
        tmp(1).name = 'motion';
        tmp(1).rdm = modelRDMs(3).rdm(1:params.analysis.n_stimdir^2,1:params.analysis.n_stimdir^2);
        tmp(2).name = 'colour';
        tmp(2).rdm = modelRDMs(3).rdm(params.analysis.n_stimdir^2+1:end,params.analysis.n_stimdir^2+1:end);
        modelRDMs = tmp;
        % todo within task rsa        
        betas = nan(length(results),2,2);
        for monk = 1:length(results)            
                % motion task
                betas(monk,1,:) = aux_regress_rdm(modelRDMs,squeeze(results(monk).rdm(1:params.analysis.n_stimdir^2,1:params.analysis.n_stimdir^2)));
                % colour task
                betas(monk,2,:) = aux_regress_rdm(modelRDMs,squeeze(results(monk).rdm(params.analysis.n_stimdir^2+1:end,params.analysis.n_stimdir^2+1:end)));
        end
        cd(params.dir.results);
        save('betas_rsa_withintasks.mat','betas');
        cd(params.dir.project);
    else
        betas = nan(length(results),length(modelRDMs));
        for monk = 1:length(results)
            betas(monk,:) = aux_regress_rdm(modelRDMs,results(monk).rdm);
        end
        cd(params.dir.results);
        save('betas_rsa.mat','betas');
        cd(params.dir.project);
    end
end