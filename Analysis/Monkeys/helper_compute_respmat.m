
function respmat = helper_compute_respmat(unitstruct,params,t_window,do_shuff,whichneurons)
    %% helper_compute_respmat(unitstruct,params,t_window,do_shuff)
    %
    % computes response matrix, based on script by Chris
    % 
    % Timo Flesch, 2021

   
    if ~exist('do_shuff')
        do_shuff = false;
    end
    if ~exist('whichneurons')
        whichneurons = 'all';
    end 

    % respmat = NaN(length(unitstruct), params.analysis.n_stimdir,params.analysis.n_stimcol,params.analysis.n_stimctx);
    % collect responses 
    uidx = 1;
    for d = 1:length(unitstruct)
        unit = unitstruct(d).unit;                        
        if ~strcmp(unit.name,'LOL')
            ud = unique(unit.task_variable.stim_dir);
            uc = unique(unit.task_variable.stim_col);
            cc = unique(unit.task_variable.context);
            if strcmp(whichneurons,'all')
                if do_shuff
                    unit = helper_shufflevars(unit);        
                end
                for u1 = 1:length(ud)
                    for u2 = 1:length(uc)
                        for c = 1:length(cc)
                            indx = find(unit.task_variable.stim_dir==ud(u1) & unit.task_variable.stim_col==uc(u2) & unit.task_variable.context==cc(c));
                            respmat(uidx,u1,u2,c) = nanmean(nanmean(unit.response(indx,t_window)));
                        end
                    end
                end
                uidx = uidx+1;
            elseif  strcmp(whichneurons,'task')
                % responses per task
                ut1 = squeeze(nanmean(unit.response(unit.task_variable.context==-1,t_window),2));
                ut2 = squeeze(nanmean(unit.response(unit.task_variable.context==1,t_window), 2));            
                % equate trial numbers 
                [ut1s,ut2s] = subsample_trials(ut1,ut2);
                % t-test on task selectivity
                [h,p] = ttest(ut1s,ut2s);
                if isnan(h)
                    h=0;
                end
                % if significant, compute respmat
                if h==1 % h1    
                    if do_shuff
                        unit = helper_shufflevars(unit);        
                    end             
                    for u1 = 1:length(ud)
                        for u2 = 1:length(uc)
                            for c = 1:length(cc)
                                indx = find(unit.task_variable.stim_dir==ud(u1) & unit.task_variable.stim_col==uc(u2) & unit.task_variable.context==cc(c));
                                respmat(uidx,u1,u2,c) = nanmean(nanmean(unit.response(indx,t_window)));
                            end
                        end
                    end
                    uidx = uidx + 1;
                end 
                % else, don't
            elseif strcmp(whichneurons,'mixed')
                % responses per task
                ut1 = squeeze(nanmean(unit.response(unit.task_variable.context==-1,t_window),2));
                ut2 = squeeze(nanmean(unit.response(unit.task_variable.context==1,t_window), 2));            
                % equate trial numbers 
                [ut1s,ut2s] = subsample_trials(ut1,ut2);
                % t-test on task selectivity
                [h,p] = ttest(ut1s,ut2s);
                % if n.s., compute respmat
                if h==0 
                    if do_shuff
                        unit = helper_shufflevars(unit);        
                    end
                    for u1 = 1:length(ud)
                        for u2 = 1:length(uc)
                            for c = 1:length(cc)
                                indx = find(unit.task_variable.stim_dir==ud(u1) & unit.task_variable.stim_col==uc(u2) & unit.task_variable.context==cc(c));
                                respmat(uidx,u1,u2,c) = nanmean(nanmean(unit.response(indx,t_window)));
                            end
                        end
                    end
                    uidx = uidx +1;
                end
            end
        end
    end
end


function unit = helper_shufflevars(unit);
    fns = fieldnames(unit.task_variable);
    n_trials = length(unit.task_variable.(fns{1}));
    % use same shufle indices for all variables, otherwise missing data
    shuffidces = randperm(n_trials);
    for ii = 1:length(fns)
        unit.task_variable.(fns{ii}) = unit.task_variable.(fns{ii})(shuffidces);
    end
end


function [ut1s, ut2s] = subsample_trials(ut1,ut2)
    [x,idx] = min([length(ut1),length(ut2)]);
    idces = randperm(x);
    ut1s = ut1(idces);
    ut2s = ut2(idces);
end
