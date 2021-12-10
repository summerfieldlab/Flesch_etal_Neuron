function mante_disp_regressionresults(params,betas_monks,betas_perm,withinTasks)
    %% mante_disp_regressionresults()
    %
    % displays results of rsa regression
    % 
    % Timo Flesch, 2020

    if ~exist('withinTasks')
        withinTasks = 0;
    end
    cols = {[52, 116, 235]./255,[161, 53, 232]./255};
    sem = @(x,dim) nanstd(x,0,dim)/sqrt(length(x));
    
    if ~withinTasks 
        modelrdms = mante_gen_modelrdms(params);
        modelnames = {modelrdms.name};


        figure();set(gcf,'Color','w');
        for ii = 1:1
            % subplot(1,2,ii);
            % beta coefficients for orig data
            hb = bar(betas_monks(ii,:));
            hb.EdgeColor = 'None';
            title(['\rm Model RSA, Monkey ' params.analysis.monknames{ii}]);
            % beta coefficients for randomised data 
            hold on;
            for jj = 1:length(modelnames)
                hs = scatter(jj.*ones(size(betas_perm,2),1),betas_perm(ii,:,jj),10,'MarkerFaceColor',[1 1 1],'MarkerEdgeColor',[.7 .7 .7]);
                he = errorbar(jj,mean(squeeze(betas_perm(ii,:,jj))),2*std(transpose(squeeze(betas_perm(ii,:,jj))),0,1),'LineWidth',1,'Color','k');                
            end
            legend([hb,hs,he],{'orig data','shuffled data','+- 2std'},'Location','NorthWest')
            legend boxoff;
            set(gca,'XTickLabel',modelnames);
            set(gca,'XTickLabelRotation',40);
            ylabel('\beta estimate (a.u.)');
            xlabel('Model RDM');
            box off;
            ylim([-.4,.7])
        end

        % set(gcf,'Position',[78   800   762   428]);
        % cd(params.dir.figures);
        % savefig('rsa_regression_results.fig');
        % saveas(gcf,'rsa_regression_results.svg');
        % saveas(gcf,'rsa_regression_results.png');
    else 
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
        for monk = 1:2
            subplot(1,2,monk);
            % bar plots 
            fh = bar(squeeze(betas_monks(monk,:,:)));
            fh(1).FaceColor = cols{1};
            fh(1).EdgeColor = 'None';
            fh(2).FaceColor = cols{2};
            fh(2).EdgeColor = 'None';
            hold on;
            % estimated null distribution
            for t = 1:2
                % first model rdm
                scatter((t-0.15).*(ones(size(betas_perm,2),1)),betas_perm(monk,:,t,1),10,'MarkerFaceColor',[.8 .8 .8],'MarkerEdgeColor',[.3 .3 .3]);
                errorbar(t-0.1,mean(betas_perm(monk,:,t,1),2),2*std(betas_perm(monk,:,t,1),0,2),'LineWidth',1,'Color','k');
                % second model rdm
                scatter((t+0.15).*(ones(size(betas_perm,2),1)),betas_perm(monk,:,t,2),10,'MarkerFaceColor',[.8 .8 .8],'MarkerEdgeColor',[.3 .3 .3]);
                errorbar(t+0.1,mean(betas_perm(monk,:,t,2),2),2*std(betas_perm(monk,:,t,2),0,2),'LineWidth',1,'Color','k');
            end
            title(['\rm Model RSA, Within Tasks, Monkey ' params.analysis.monknames{monk}]);
            set(gca,'XTickLabel',tasks);
            xlabel('Task');
            ylabel('\beta estimate');
            box off;
            legend([fh(1), fh(2)],{[tasks{1} ' model rdm'], [tasks{2} ' model rdm']},'Location','SouthWest');
            legend boxoff;
            ylim([-.5,1.0]);
            
            
        end
        set(gcf,'Position',[995, 857, 1259, 465]);
        cd(params.dir.figures);
        savefig('rsa_regression_within_results.fig');
        saveas(gcf,'rsa_regression_within_results.svg');
        saveas(gcf,'rsa_regression_within_results.png');

end