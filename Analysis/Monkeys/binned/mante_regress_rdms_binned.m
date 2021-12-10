function betas = mante_regress_rdms_binned(params,results,withinTasks)
    %% mante_regress_rdms_binned(results)
    % 
    % performs regression on early, middle and late time bins
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
        betas = nan(length(results),2,2,size(results(1).rdms,1));
        for monk = 1:length(results)
            for bid = 1:size(results(monk).rdms,1)
                % motion task
                betas(monk,1,:,bid) = aux_regress_rdm(modelRDMs,squeeze(results(monk).rdms(bid,1:params.analysis.n_stimdir^2,1:params.analysis.n_stimdir^2)));
                % colour task
                betas(monk,2,:,bid) = aux_regress_rdm(modelRDMs,squeeze(results(monk).rdms(bid,params.analysis.n_stimdir^2+1:end,params.analysis.n_stimdir^2+1:end)));
            end
        end
        cd(params.dir.results);
        save('betas_rsa_withintasks_binned.mat','betas');
        cd(params.dir.project);
    else
        betas = nan(length(results),length(modelRDMs),size(results(1).rdms,1));
        for monk = 1:length(results)
            for bid = 1:size(results(monk).rdms,1)
                betas(monk,:,bid) = aux_regress_rdm(modelRDMs,squeeze(results(monk).rdms(bid,:,:)));
            end
        end
        cd(params.dir.results);
        save('betas_rsa_binned.mat','betas');
        cd(params.dir.project);
    end
end