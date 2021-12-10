function fmri_plot_taskswitching_results()
    %% fmri_plot_taskswitching_results()
    % 
    % plots analysis of task switching effects 
    %
    % Timo Flesch, 2020
    % Human Information Processing lab 
    % University of Oxford 

    %% parameters

    % bad subjects 
    badsubs_total = [19,28];
    % bad subjects not excluded from behav yet:
    badsubs_behav = 19;

    % ROIs of interest
    rsa_rois = {'ROI_RIGHTANGULARGYRUS_x=36_y=-49_z=42_170voxels_Sphere12', ...
    'ROI_RIGHTIFG_x=43_y=14_z=1.750000e+01_161voxels_Sphere12'};    
    rsa_roi_labels ={'Angular Gyrus','IFG'};

    univar_rois = {'clustermask_switchstay_BA7.nii', ...
                'clustermask_switchstay_leftIFG.nii', ...
                'clustermask_switchstay_superiormedial.nii'};
    univar_roi_labels ={'BA7', 'leftIFG','ACC'};

    % directories     
    % behaviour 
    dir_behav_scan = 'project/results_paper/behav/'; % acc_all_scan_granada.mat
    dir_behav_baseline = 'projectexp_2_granada_training/results/matfiles/'; % acc_all_training_granada
    dir_behav_arena = 'projectexp_1_granada_arenatask/results/matfiles/'; %tauas_features_exp1_granada_32subs.mat
    % rsa 
    dir_rsa = 'project/results_paper/GLM_3_RSA/GROUP/';
    % univar 
    dir_univar = 'project/results_paper/glm_1_switchStay_cueLock/';
    % masks 
    dir_masks = 'project/results_paper/masks/';
    



    % prepare data for analysis 
    load([dir_behav_baseline 'acc_all_training_granada.mat'])
    acc_baseline = acc_all;
    load([dir_behav_scan 'acc_all_scan_granada.mat']);
    acc_scan = acc_all;

    load([dir_behav_baseline 'rt_all_training_granada.mat'])
    rt_baseline = rt_all;
    load([dir_behav_scan 'rt_all_scan_granada.mat']);
    rt_scan = rt_all;

    acc_scan.taskSwitchStay.stay(badsubs_total) = [];
    acc_scan.taskSwitchStay.switch(badsubs_total) = [];
    accdiff_scan = acc_scan.taskSwitchStay.stay-acc_scan.taskSwitchStay.switch;
    accdiff_baseline = acc_baseline.taskSwitchStay.p2.stay-acc_baseline.taskSwitchStay.p2.switch;
    
    rt_scan.taskSwitchStay.stay(badsubs_total) = [];
    rt_scan.taskSwitchStay.switch(badsubs_total) = [];
    rtdiff_scan = rt_scan.taskSwitchStay.switch-rt_scan.taskSwitchStay.stay;
    rtdiff_baseline = rt_baseline.taskSwitchStay.p2.switch-rt_baseline.taskSwitchStay.p2.stay;
    
    
    %% 1. SWITCH COST (ACC & RT)
    helper_plot_switchvsstay(acc_baseline,acc_scan,'accuracy')
    helper_plot_switchvsstay(rt_baseline,rt_scan,'reaction time')

    %% 2a. NEURAL SWITCH COST x BEHAV SWITCH COST (ACC & RT)
    neural_switchcost = fmri_taskswitching_extract_betas()
    % 1. Accuracy
    for ii = 1:length(univar_roi_labels)
        helper_scatter_costdiff(neural_switchcost.betas(ii,:)',accdiff_scan,1,'scan',['Accuracy ROI ' univar_roi_labels{ii}],'Neural Switch Cost','Behav Switch Cost','kendall_switchcost_brainbehav')
    end 
    % 2. Reaction Time 
    for ii = 1:length(univar_roi_labels)
        helper_scatter_costdiff(neural_switchcost.betas(ii,:)',rtdiff_scan,1,'scan',['Reaction Time ROI ' univar_roi_labels{ii}],'Neural Switch Cost','Behav Switch Cost','kendall_switchcost_brainbehav')
    end 

    % %% 2b GRID PRIOR X NEURAL SWITCH COST 
    load([dir_behav_arena 'tauas_features_exp1_granada_32subs.mat'])
    gridprior = tauas_features(:,3);
    gridprior(badsubs_total) = [];
    for ii = 1:length(univar_roi_labels)
        helper_scatter_costdiff(gridprior,neural_switchcost.betas(ii,:)',1,'scan',['Grid Prior ROI ' univar_roi_labels{ii}],'Grid Prior','Neural Switch Cost','kendall_switchcost_braingridprior')
    end 

    %% 2c GRID PRIOR X BEHAV SWITCH COST (ACC & RT)
    % 1. Accuracy
    
    helper_scatter_costdiff(gridprior,accdiff_scan,1,'scan',['Accuracy'] ,'Grid Prior','Behav Switch Cost','kendall_switchcost_gridpriorbehav')
    
    % 2. Reaction Time 
    
    helper_scatter_costdiff(gridprior,rtdiff_scan,1,'scan',['Reaction Time'] ,'Grid Prior','Behav Switch Cost','kendall_switchcost_gridpriorbehav')
    
    
    % %% 3a. NEURAL TASK FACTORISATION x SWITCH COST (ACC & RT)
    % Acc
    for rid = 1:length(rsa_rois)
        load([dir_rsa 'groupAvg_modelBetas_cval_slRSA_paper__' rsa_rois{rid} '.mat']);
        c = results.corrs(:,3);
        c(badsubs_behav) = [];
        helper_scatter_costdiff(c,accdiff_scan,1,'scan',['Accuracy ROI ' rsa_roi_labels{rid}],'Neural Task Factorisation','Switch Cost','kendall_switchcost_factorisation');
    end
    % Reaction Time
    for rid = 1:length(rsa_rois)
        load([dir_rsa 'groupAvg_modelBetas_cval_slRSA_paper__' rsa_rois{rid} '.mat']);
        c = results.corrs(:,3);
        c(badsubs_behav) = [];
        helper_scatter_costdiff(c,rtdiff_scan,1,'scan',['Reaction Time ROI ' rsa_roi_labels{rid}],'Neural Task Factorisation','Switch Cost','kendall_switchcost_factorisation');
    end

    %% 3b BEHAV TASK FACTORISATION X SWITCH COST (ACC & RT)
    load([dir_behav_scan 'taus_3Dvs2D_scan_granada.mat'])
    taus = taus.blocked.both(:,1);
    taus(badsubs_total) = [];
    % 1. Accuracy    
    helper_scatter_costdiff(taus,accdiff_scan,1,'scan',['Accuracy'] ,'Behav Task Factorisation','Behav Switch Cost','kendall_switchcost_behavfactorisation')
    
    % 2. Reaction Time 
    
    helper_scatter_costdiff(taus,rtdiff_scan,1,'scan',['Reaction Time'] ,'Behav Task Factorisation','Behav Switch Cost','kendall_switchcost_behavfactorisation')
    

    
    %% 4 NEURAL COMPRESSION X SWITCH COST (ACC & RT)
    for rid = 1:length(rsa_rois)
        load([dir_rsa 'results_compressangle_' rsa_rois{rid} '.mat']);
        xy = [];
        for ii = 1:size(results,1)
            [~,m] = max(squeeze(results(ii,:,:)),[],'all','linear');
            [xy(ii,2),xy(ii,1)] = ind2sub([size(results,2),size(results,3)],m);
       end
        c = xy(:,2)/101;
        c(badsubs_behav) = [];        
        helper_scatter_costdiff(c,accdiff_scan,1,'scan',['Accuracy ROI ' rsa_roi_labels{rid}],'Neural Compression','Switch Cost','kendall_switchcost_compression');
    end

    for rid = 1:length(rsa_rois)
        load([dir_rsa 'results_compressangle_' rsa_rois{rid} '.mat']);
        xy = [];
        for ii = 1:size(results,1)
            [~,m] = max(squeeze(results(ii,:,:)),[],'all','linear');
            [xy(ii,2),xy(ii,1)] = ind2sub([size(results,2),size(results,3)],m);
       end
        c = xy(:,2)/101;
        c(badsubs_behav) = [];        
        helper_scatter_costdiff(c,rtdiff_scan,1,'scan',['Reaction Time ROI ' rsa_roi_labels{rid}],'Neural Compression','Switch Cost','kendall_switchcost_compression');
    end


end

function helper_scatter_costdiff(c,accdiff,do_rt,phase_id,title_label,x_label,y_label,fname)
    figure();set(gcf,'Color','w');
    
    % scatter(c,accdiff,'MarkerFaceColor',[68, 185, 227]./255, 'MarkerEdgeColor', [68, 185, 227]./255);
    scatter(c,accdiff,'MarkerFaceColor',[.8 ,.8 ,.8], 'MarkerEdgeColor', [0,0,0]);
    hold on;    
    grid off;
    box off;   
    xlabel(x_label);
    ylabel(y_label);       
    
    title([title_label]);
    disp('-----')
    if do_rt
        c = rankTransform_equalsStayEqual(c);
        accdiff = rankTransform_equalsStayEqual(accdiff);
        [r,p] = corr(c,accdiff,'Type','Kendall')   
    else 
        [r,p] = corr(c,accdiff,'Type','Pearson')   
    end

    get(gcf,'Position');
    % if do_rt 
    %     set(gca,'XTick',0:.2:1)
    %     set(gca,'YTick',0:.2:1);
    %     xlim([0,1]);
    %     ylim([0,1]);
    % end
    l = lsline;
    l.Color = 'k';
    set(gcf,'Position',[793   949   344   287]);
    
    saveas(gcf,[fname '_' replace(title_label,' ','_') '_p' replace(num2str(p),'.','_') '_r' replace(num2str(r),'.','_') '.svg'],'svg');

end 



function helper_plot_switchvsstay(acc_day1,acc_scan,titlestr)
    badsubs = [];

    sem = @(X,dim) std(X,0,dim)./sqrt(size(X,dim));


    phaseName      = {'baseline', 'scan'};
    % %% task switch -------------------------------------------------------------------------------------------------------------------
    acc_taskSwitch = {};
    acc_taskSwitch{1} = [acc_day1.taskSwitchStay.p2.stay,acc_day1.taskSwitchStay.p2.switch];
    acc_taskSwitch{2} = [acc_scan.taskSwitchStay.stay,acc_scan.taskSwitchStay.switch];
    acc_taskSwitch{2}(badsubs,:)  = [];

    colvals = linspace(0.5,0.7,2);

    figure();set(gcf,'Color','w');
    e = [sem(acc_taskSwitch{1},1);sem(acc_taskSwitch{2},1)];
    m = [mean(acc_taskSwitch{1},1);mean(acc_taskSwitch{2},1)];
    b = barwitherr(e,m);
    b(1).FaceColor = [1,1,1].*colvals(1);
    b(2).FaceColor = [1,1,1].*colvals(2);
    hold on;
    % stay trials
    plot(repmat([1,2]-0.15,30,1),[acc_taskSwitch{1}(:,1),acc_taskSwitch{2}(:,1)],'MarkerFaceColor',[.9 .9 .9],'MarkerEdgeColor',[.3 .3 .3],'LineStyle','none','Marker','o')
    hold on
    % switch trials
    plot(repmat([1,2]+0.15,30,1),[acc_taskSwitch{1}(:,2),acc_taskSwitch{2}(:,2)],'MarkerFaceColor',[.9 .9 .9],'MarkerEdgeColor',[.3 .3 .3],'LineStyle','none','Marker','o')

    plot([1-0.15; 1+0.15],[acc_taskSwitch{1}(:,1)';acc_taskSwitch{1}(:,2)'],'Color',[.8 .8 .8],'LineStyle','-');
    plot([2-0.15; 2+0.15],[acc_taskSwitch{2}(:,1)';acc_taskSwitch{2}(:,2)'],'Color',[.8 .8 .8],'LineStyle','-');

    eb = errorbar(1-0.15,mean(acc_taskSwitch{1}(:,1)),sem(acc_taskSwitch{1}(:,1),1),'LineWidth',2.5);
    eb.Color = [0,0,0];
    eb = errorbar(1+0.15,mean(acc_taskSwitch{1}(:,2)),sem(acc_taskSwitch{1}(:,2),1),'LineWidth',2.5);
    eb.Color = [0,0,0];
    eb = errorbar(2-0.15,mean(acc_taskSwitch{2}(:,1)),sem(acc_taskSwitch{2}(:,1),1),'LineWidth',2.5);
    eb.Color = [0,0,0];
    eb = errorbar(2+0.15,mean(acc_taskSwitch{2}(:,2)),sem(acc_taskSwitch{2}(:,2),1),'LineWidth',2.5);
    eb.Color = [0,0,0];
    pvals = [];
    [~,pvals(1),~,s1] = ttest(acc_taskSwitch{1}(:,1),acc_taskSwitch{1}(:,2));
    [~,pvals(2),~,s2] = ttest(acc_taskSwitch{2}(:,1),acc_taskSwitch{2}(:,2));
    [~,pvals(3),~,s3] = ttest(acc_taskSwitch{1}(:,1)-acc_taskSwitch{1}(:,2),acc_taskSwitch{2}(:,1)-acc_taskSwitch{2}(:,2));

    pvals
    s1
    s2
    s3
    sigstar({[0.85,1.15],[1.85,2.15],[1,2]},pvals)
    box off;
    set(gca,'XTick',1:length(acc_taskSwitch));
    set(gca,'XTickLabel',phaseName);
    set(gca,'TickDir','out');
    xlim([0,length(acc_taskSwitch)+1]);
    if strcmp(titlestr,'accuracy')
        ylabel('\bf Accuracy (%)');
        set(gca,'YTick',[0:.2:1])
        set(gca,'YTickLabel',[0:20:100])
        ylim([0.3,1.2]);
        plot(get(gca,'XLim'),[.5 .5],'k--');
    elseif strcmp(titlestr,'reaction time')
        ylabel('\bf RT (s)');
    end
    set(gcf,'Position',[680   546   503   425]);
    legend([b(1),b(2)],{'Task Stay','Task Switch'},'Location','NorthEastOutside')
    legend('boxoff');
    title([upper(titlestr(1)), titlestr(2:end)]);
end