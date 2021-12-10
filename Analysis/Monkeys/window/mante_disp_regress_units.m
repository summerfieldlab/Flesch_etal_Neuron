function [allbetas_task,allbetas_mixed] = mante_disp_regress_units(params,results,stats)
    %% mante_disp_regress_units(params,results,stats)
    %
    % displays several summary statistic of estimated 
    % single unit activty patterns
    %
    % Timo Flesch, 2021

    
    sem = @(x,dim) nanstd(x,0,dim)/sqrt(length(x));
    
    thresh = 0.01;

    %% 1. fraction of task-sensitive units 
    figure(); set(gcf,'Color','w')
    cols = {[ 0    0.4470    0.7410],[0.8500    0.3250    0.0980],[0.9290    0.6940    0.1250], [.8 0 0], [0 .4 0],[.2 .2 .2]};
    for monk = 1:2
        respmat = results(monk).respmat;
        p = stats(monk).pvals;
        [pfdr,~] = fdr(p(:),thresh);
        p = p < pfdr;
        % p = p < 0.001;
        nsig = sum(any(p,2));
        disp(['n significant ' num2str(nsig)])
        idx_t1 =  (p(:,1)==0 & p(:,2)==1) & (p(:,3)==0 & p(:,4)==0);
        idx_t2 =  (p(:,1)==0 & p(:,2)==0) & (p(:,3)==1 & p(:,4)==0);

        idx_irt1 =  (p(:,1)==1 & p(:,2)==0) & (p(:,3)==0 & p(:,4)==0);
        idx_irt2 =  (p(:,1)==0 & p(:,2)==0) & (p(:,3)==0 & p(:,4)==1);
        
        idx_mix = (p(:,1)==1 & p(:,2)==1) | (p(:,3)==1 & p(:,4)==1);
       
        idx_col = (p(:,1)==1 & p(:,2)==0) & (p(:,3)==1 & p(:,4)==0);
        idx_mot = (p(:,1)==0 & p(:,2)==1) & (p(:,3)==0 & p(:,4)==1);


        task_fract = [sum(idx_t1)/nsig,sum(idx_t2)/nsig,sum(idx_irt1)/nsig, sum(idx_irt2)/nsig,sum(idx_mix)/nsig,sum(idx_col)/nsig,sum(idx_mot)/nsig];
        task_fract = [task_fract,1-sum(task_fract)];
        bh = bar(monk,task_fract,'stacked');
        hold on;
        for ii = 1:length(cols)
            bh(ii).EdgeColor = 'None';
            bh(ii).FaceColor = cols{ii};
        end
    end
    box off;
    xlabel(' monkey')
    set(gca,'XTick',1:2);
    set(gca,'XTickLabel',params.analysis.monknames);
    ylabel('Percentage of Units');
    ylim([0,1]);
    set(gca,'YTickLabel',get(gca,'YTick').*100)
    legend(bh,{'task A','task B', 'taskAirrel','taskBirrel','both dimensions', 'only colour','only motion','mixed'},'Box','off');
    title('\rmEstimated Feature Selectivity of Recording Units')
    ylim([0,1]);



    %% 2. response matrices
    
    for monk = 1:2
        respmat = results(monk).respmat;
        p = stats(monk).pvals;
        [pfdr,~] = fdr(p(:),thresh);
        disp(['fdr thresh ' num2str(pfdr)]);
        p = p < pfdr;
        
        
        idx_t1 =  (p(:,1)==1 | p(:,2)==1) & (p(:,3)==0 & p(:,4)==0);
        idx_t2 =  (p(:,1)==0 & p(:,2)==0) & (p(:,3)==1 | p(:,4)==1);
        idx_mix = (p(:,1)==1 & p(:,2)==1) | (p(:,3)==1 & p(:,4)==1);

        figure(); set(gcf,'Color','w')

        r_t1 = scale01(reshape(mean(zscore(respmat(idx_t1,1:36),[],2),1),[6,6]));
        r_t2 = scale01(reshape(mean(zscore(respmat(idx_t2,37:end),[],2),1),[6,6]));
        r_mix1 = scale01(reshape(mean(zscore(respmat(idx_mix,1:36),[],2),1),[6,6]));
        r_mix2 = scale01(reshape(mean(zscore(respmat(idx_mix,37:end),[],2),1),[6,6]));
        
        subplot(2,2,1);
        imagesc(r_t1);
        title({'\rm Motion Task - Motion Task selective units';['Monkey ' params.analysis.monknames{monk}]});
        axis square;
        xlabel('motion');
        ylabel('colour');
        colormap('viridis');
        % set(gca,'CLim',[0.01,0.03])
        colorbar()
        subplot(2,2,2);
        imagesc(r_t2);
        title({'\rm Colour Task - Colour Task selective units';['Monkey ' params.analysis.monknames{monk}]});
        axis square;
        xlabel('motion');
        ylabel('colour');
        colormap('viridis');
        % set(gca,'CLim',[0.01,0.03])
        colorbar()
        subplot(2,2,3)
        imagesc(r_mix1);
        title({'\rm Motion Task - Stimulus selective units';['Monkey ' params.analysis.monknames{monk}]});
        axis square;
        xlabel('motion');
        ylabel('colour');
        colormap('viridis');
        % set(gca,'CLim',[0.01,0.03])
        colorbar()
        subplot(2,2,4);
        imagesc(r_mix2);
        title({'\rm Colour Task - Stimulus selective units';['Monkey ' params.analysis.monknames{monk}]});
        axis square;
        xlabel('motion');
        ylabel('colour');
        colormap('viridis');
        % set(gca,'CLim',[0.01,0.03])
        colorbar()
    end
    
    %% 3. Regression on model RDMs 
    modCol   = mante_gen_behavmodelrdms(1);
    modMotion = mante_gen_behavmodelrdms(2);
    modDiag   = mante_gen_behavmodelrdms(3); % diag 1 is interaction
    
    rdm3Dmod   = squareform(pdist([modMotion.choiceMat(:);modCol.choiceMat(:)]));
    rdm2Dmod   = squareform(pdist([modDiag.choiceMat(:);modDiag.choiceMat(:)]));
    dmat = [];
    dmat(:,1) = zscore(vectorizeRDM(rdm3Dmod));
    dmat(:,2) = zscore(vectorizeRDM(rdm2Dmod));
    
    dmat = [ones(size(dmat,1),1), dmat];

    for monk = 1:2
        respmat = results(monk).respmat;
        p = stats(monk).pvals;
        [pfdr,~] = fdr(p(:),thresh);
        p = p < pfdr;
        % p = p < 0.001;
        % nsig = sum(any(p,2));
        % disp(['n significant ' num2str(nsig)])
        
        idx_t1 =  (p(:,1)==1 | p(:,2)==1) & (p(:,3)==0 & p(:,4)==0);
        idx_t2 =  (p(:,1)==0 & p(:,2)==0) & (p(:,3)==1 | p(:,4)==1);      
        % idx_mix = ~(idx_t1+idx_t2);
        idx_mix = (p(:,1)==1 & p(:,2)==1) | (p(:,3)==1 & p(:,4)==1);

       
        
        rmat = respmat(idx_t1,1:36);
        nu = size(rmat,1);        
        r_t1 = reshape(rmat,[nu,6,6]);

        rmat = respmat(idx_t2,37:end);
        nu = size(rmat,1);
        r_t2 = reshape(rmat,[nu,6,6]);

        rmat = respmat(idx_mix,1:36);
        nu = size(rmat,1);        
        r_mix1 = reshape(rmat,[nu,6,6]);

        rmat = respmat(idx_mix,37:end);
        nu = size(rmat,1);        
        r_mix2 = reshape(rmat,[nu,6,6]);
        betas = [];
        betas_task = [];
        betas_mixed = [];
        for ii = 1:min([size(r_t1,1),size(r_t2,1)])
            r1 = squeeze(r_t1(ii,:,:));
            r1 = zscore(r1(:));
            r2 = squeeze(r_t2(ii,:,:));
            r2 = zscore(r2(:));
            b = regress(zscore(vectorizeRDM(squareform(pdist([r1;r2]))))',dmat);
            %keyboard
            betas_task(ii,:) = b(2:end);
        end
        for ii = 1:min([size(r_mix1,1),size(r_mix2,1)])
            m1 = squeeze(r_mix1(ii,:,:));
            m1 = zscore(m1(:));
            m2 = squeeze(r_mix2(ii,:,:));
            m2 = zscore(m2(:));
            b = regress(zscore(vectorizeRDM(squareform(pdist([m1;m2]))))',dmat);
            betas_mixed(ii,:) = b(2:end);
        end

        betas_mu = [squeeze(mean(betas_task,1));squeeze(mean(betas_mixed,1))];
        betas_err = [squeeze(std(betas_task,0,1)./(sqrt(size(betas_task,1))));squeeze(std(betas_mixed,0,1)./(sqrt(size(betas_mixed,1))))];
        figure();set(gcf,'Color','w');
        barwitherr(squeeze(betas_err(:,:)),squeeze(betas_mu(:,:)));
        box off
        set(gca,'XTickLabel',{'task selective', 'mixed selective'});
        xlabel('units')
        ylabel('\beta estimate')
        set(gcf,'Color','w')
        legend({'2D model','1D model'},'Box','off')
        title('\rm Model RSA on average unit activity');
        % keyboard
        allbetas_task{monk} = betas_task;
        allbetas_mixed{monk} = betas_mixed;
    end




end

    