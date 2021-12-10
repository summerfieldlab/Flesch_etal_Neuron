function results = mante_regress_singleunits(params)
    %% mante_regress_singleunits()
    %
    % tests for context-specific feature selectivity 
    % at single unit level
    %    
    % Timo Flesch, 2021
    
    
    results = struct();
    for monk = 1:params.analysis.n_monks
        cd([params.dir.data 'monkey' num2str(monk) '/']);
        allunits = dir('*.mat');                      
        results(monk).betas = [];  % betas for motion_in_ctxt_1, color_in_ctxt_1, motion_in_ctxt_2, color_in_ctxt_2.
        results(monk).pvals = []; % associated p values
        
        for d = 1:length(allunits)
            load(allunits(d).name);            
            
            y = zscore(squeeze(nanmean(unit.response(:,218:651),2)));            
            
            dmat = [unit.task_variable.stim_dir,unit.task_variable.stim_col,unit.task_variable.stim_dir,unit.task_variable.stim_col];
            % motion in ctx 1, hence set motion in ctx 2 to zero
            dmat(unit.task_variable.context==1,1) = 0;
            % ditto for color
            dmat(unit.task_variable.context==1,2) = 0;
            % same for context 2
            dmat(unit.task_variable.context==-1,3) = 0;
            % ditto for color
            dmat(unit.task_variable.context==-1,4) = 0;
            % standardise variables
            dmat = zscore(dmat,1);
            % regress 
            [b,~,s] = glmfit(dmat,y);
            
            
            results(monk).betas(d,:) = b(2:end);
            results(monk).pvals(d,:) = s.p(2:end);
            
        end        
    end
    disp('...done');
end


function [ut1s, ut2s] = subsample_trials(ut1,ut2)
    [x,idx] = min([length(ut1),length(ut2)]);
    idces = randperm(x);
    ut1s = ut1(idces);
    ut2s = ut2(idces);
end