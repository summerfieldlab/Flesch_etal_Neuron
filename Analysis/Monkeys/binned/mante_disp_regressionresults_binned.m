function mante_disp_regressionresults_binned(params, betas_monks,betas_perm, withinTasks)
    %%  mante_disp_regressionresults_binned(params, betas_monks,betas_perm, withinTasks)
    % 
    % displays rsa regression results for early, middle and late time windows
    % x axis: time window
    % bar-colour: model rdm 
    %
    % Timo Flesch, 2021

    if ~exist('withinTasks')
        withinTasks = false;
    end
    
    if withinTasks==0
        % cols = {'magenta','green','blue','cyan'};
        modelrdms = mante_gen_modelrdms(params);
        modelnames = {modelrdms.name};
        figure(1);set(gcf,'Color','w');
        % one subplot per monkey
        for monk =1:2
            subplot(1,2,monk);
            % plot betas (transpose such that rows: time bins, cols: modelrdms)            
            bh = bar(transpose(squeeze(betas_monks(monk,:,:))));
            for ii =1:length(bh)
                bh(ii).EdgeColor = 'None';
                % bh(ii).FaceColor = cols{ii};                
            end
            hold on;
            for modi = 1:length(modelnames)
                for t=1:3
                    scatter((bh(modi).XEndPoints(t)).*(ones(size(betas_perm,2),1)),betas_perm(monk,:,modi,t),10,'MarkerFaceColor',[1 1 1],'MarkerEdgeColor',[.7 .7 .7]);                    
                end
                errorbar(bh(modi).XEndPoints,squeeze(mean(betas_perm(monk,:,modi,:),2)),squeeze(2*std(betas_perm(monk,:,modi,:),0,2)),'LineWidth',1,'Color','k','LineStyle','None');
            end
            set(gca,'XTickLabel',{'early','middle','late'});
            xlabel('time window (217ms each)');
            ylabel('\beta estimate (a.u.)');          
            box off
            ylim([-0.4,0.8]);
            title(['\rm Model RSA, Monkey ' params.analysis.monknames{monk}]);
            legend([bh],modelnames,'Box','off','Location','NorthWest');
            
        end
    
    else 
        cols = {[52, 116, 235]./255,[161, 53, 232]./255}; 
        modelrdms = mante_gen_modelrdms(params);
        tmp = struct();
        tmp(1).name = 'motion';
        tmp(1).rdm = modelrdms(3).rdm(1:params.analysis.n_stimdir^2,1:params.analysis.n_stimdir^2);
        tmp(2).name = 'colour';
        tmp(2).rdm = modelrdms(3).rdm(params.analysis.n_stimdir^2+1:end,params.analysis.n_stimdir^2+1:end);
        modelrdms = tmp;
        modelnames = {modelrdms.name};

        figure(); set(gcf,'Color','w');
        fidx = 1;
        monks = params.analysis.monknames;
        tasks = modelnames;
        idx = 1
        for task = 1:2
            for monk = 1:2
                subplot(2,2,idx);
                % bar plots 
                bh = bar(transpose(squeeze(betas_monks(monk,task,:,:))));                
                for ii = 1:length(bh)
                    bh(ii).EdgeColor = 'None';
                    bh(ii).FaceColor = cols{ii};                    
                end
                hold on;
                for t=1:3
                    scatter((t-0.15).*(ones(size(betas_perm,2),1)),betas_perm(monk,:,task,1,t),10,'MarkerFaceColor',[1 1 1],'MarkerEdgeColor',[.7 .7 .7]);
                    scatter((t+0.15).*(ones(size(betas_perm,2),1)),betas_perm(monk,:,task,2,t),10,'MarkerFaceColor',[1 1 1],'MarkerEdgeColor',[.7 .7 .7]);
                end
                errorbar([1,2,3]-0.1,squeeze(mean(betas_perm(monk,:,task,1,:),2)),squeeze(2*std(betas_perm(monk,:,task,1,:),0,2)),'LineWidth',1,'Color','k','LineStyle','None');
                % second model rdm
                errorbar([1,2,3]+0.1,squeeze(mean(betas_perm(monk,:,task,2,:),2)),squeeze(2*std(betas_perm(monk,:,task,2,:),0,2)),'LineWidth',1,'Color','k','LineStyle','None');
                set(gca,'XTickLabel',{'early','middle','late'});
                xlabel('time window (217ms each)');
                ylabel('\beta estimate (a.u.)');          
                box off
                title(['\rm Model RSA, Monkey ' params.analysis.monknames{monk} ', ' tasks{task} ' task']);
                legend([bh(1), bh(2)],{[tasks{1} ' model rdm'], [tasks{2} ' model rdm']},'Location','NorthWest','Box','off');
                idx = idx +1;
                ylim([-0.2,1]);
            end
        end
        set(gcf,'Position',[995,  714, 1328,  608]);
    end