function betas = mante_regress_perm_rdms_svd(params,results,withinTasks)
    %% mante_regress_perm_rdms_svd(results)
    % 
    % performs regression in a sliding time window
    % on null distribution data
    % 
    % Timo Flesch, 2020

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
        betas = nan(length(results),params.stats.n_perms,2,2,size(results(1).rdms,2));
        for monk = 1:length(results)
            for perm = 1:params.stats.n_perms
                for sv = 1:size(results(monk).rdms,2)
                    % motion task
                    betas(monk,perm,1,:,sv) = aux_regress_rdm(modelRDMs,squeeze(results(monk).rdms(perm,sv,1:params.analysis.n_stimdir^2,1:params.analysis.n_stimdir^2)));
                    % colour task
                    betas(monk,perm,2,:,sv) = aux_regress_rdm(modelRDMs,squeeze(results(monk).rdms(perm,sv,params.analysis.n_stimdir^2+1:end,params.analysis.n_stimdir^2+1:end)));
                end
            end
        end
        cd(params.dir.results);
        save('betas_perm_rsa_withintasks_svd.mat','betas');
        cd(params.dir.project);
    else
        betas = nan(length(results),params.stats.n_perms,length(modelRDMs),size(results(1).rdms,2));
        for monk = 1:length(results)
            for perm = 1:params.stats.n_perms
                for sv = 1:size(results(monk).rdms,2)
                    betas(monk,perm,:,sv) = aux_regress_rdm(modelRDMs,squeeze(results(monk).rdms(perm,sv,:,:)));
                end
            end
        end
        cd(params.dir.results);
        save('betas_perm_rsa_svd.mat','betas');
        cd(params.dir.project);
    end
end