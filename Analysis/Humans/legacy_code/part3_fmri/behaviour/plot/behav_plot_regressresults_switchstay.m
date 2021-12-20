function behav_plot_regressresults_switchstay()

    sem = @(X,dim) std(X,0,dim)./sqrt(size(X,dim));

    % switch betas
    load('rdmSet_Switch_scan_granada.mat')
    betas = rsa_regress3Dvs2DModelBehavRDMs(rdmSet);
    betas_switch = betas.blocked.both;

    % load stay betas 
    load('rdmSet_Stay_scan_granada.mat')
    betas = rsa_regress3Dvs2DModelBehavRDMs(rdmSet);
    betas_stay = betas.blocked.both;

    mu_betas = transpose([mean(betas_stay,1);mean(betas_switch,1)]);
    er_betas = transpose([sem(betas_stay,1);sem(betas_switch,1)]);

    bh = bar(mu_betas);
    hold on;
    scatter(ones(size(betas_stay,1),1).*0.8,betas_stay(:,1),'MarkerEdgeColor','k','MarkerFaceColor',[.8 .8 .8]);
    scatter(ones(size(betas_switch,1),1).*1.1,betas_switch(:,1),'MarkerEdgeColor','k','MarkerFaceColor',[.8 .8 .8]);
    scatter(ones(size(betas_stay,1),1).*1.8,betas_stay(:,2),'MarkerEdgeColor','k','MarkerFaceColor',[.8 .8 .8]);
    scatter(ones(size(betas_switch,1),1).*2.1,betas_switch(:,2),'MarkerEdgeColor','k','MarkerFaceColor',[.8 .8 .8]);
    plot([0.8,1.1],[betas_stay(:,1),betas_switch(:,1)],'Color',[.8 .8 .8]);
    plot([1.8,2.1],[betas_stay(:,2),betas_switch(:,2)],'Color',[.8 .8 .8]);
    errorbar([0.84,1.14;1.84,2.14],mu_betas,er_betas,'LineStyle','none','Color','k','LineWidth',2)