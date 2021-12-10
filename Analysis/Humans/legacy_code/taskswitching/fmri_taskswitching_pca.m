function [X,labels,b1]= fmri_taskswitching_pca()
    %% fmri_taskswitching_pca()
    %
    % performs pca on various behavioural and neural measures
    % goal: test if there is any relationship at all between task factorisation and switch cost
    % approach:
    % 1. plot covariance matrix 
    % 2. pca of scores and loadings + scree plot 
    % 3. visualisation of data in 2D and 3D (1st two and three PCs respectively)
    %
    % Timo Flesch, 2020

    %%TODO
    % add 
    % - behav model corrs: diag model 
    % - sigmoid fits: intrusion irreldim


    rsa_rois = {'ROI_RIGHTANGULARGYRUS_x=36_y=-49_z=42_170voxels_Sphere12', ...
    'ROI_RIGHTIFG_x=43_y=14_z=1.750000e+01_161voxels_Sphere12'};    
    rsa_roi_labels ={'Angular Gyrus','IFG'};

    univar_rois = {'clustermask_switchstay_BA7.nii', ...
                'clustermask_switchstay_leftIFG.nii', ...
                'clustermask_switchstay_superiormedial.nii'};
    univar_roi_labels ={'BA7', 'leftIFG','ACC'};

    %% directories     
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
    


    % bad subjects 
    badsubs_total = [19,28];
    badsubs_behav = 19;

    %% load data and put in measure-x-subject matrix 
    X = [];
    labels = {};

    % - neural switch cost 
    neural_switchcost = fmri_taskswitching_extract_betas()
    %   - neural switch cost BA7
    %   - neural switch cost leftIFG
    %   - neural switch cost ACC
    X(1:3,:) = neural_switchcost.betas;
    labels{1} = 'sBA7';
    labels{2} = 'sLIFG';
    labels{3} = 'sMFG';
    
    % - acc switch cost 
    load([dir_behav_scan 'acc_all_scan_granada.mat']);
    acc_scan = acc_all;    
    acc_scan.taskSwitchStay.stay(badsubs_total) = [];
    acc_scan.taskSwitchStay.switch(badsubs_total) = [];
    X(4,:) = (acc_scan.taskSwitchStay.stay-acc_scan.taskSwitchStay.switch)./acc_scan.taskSwitchStay.stay+acc_scan.taskSwitchStay.switch;
    labels{4} = 'sACC';
    % - RT switch cost 
    load([dir_behav_scan 'rt_all_scan_granada.mat']);
    rt_scan = rt_all;
    rt_scan.taskSwitchStay.stay(badsubs_total) = [];
    rt_scan.taskSwitchStay.switch(badsubs_total) = [];
    X(5,:) = (rt_scan.taskSwitchStay.switch-rt_scan.taskSwitchStay.stay)./(rt_scan.taskSwitchStay.switch+rt_scan.taskSwitchStay.stay);
    labels{5} = 'sRT';
    
    % - grid prior 
    load([dir_behav_arena 'tauas_features_exp1_granada_32subs.mat'])
    gridprior = tauas_features(:,3);
    gridprior(badsubs_total) = [];
    X(6,:) = gridprior;
    labels{6} = 'GRIDP';

    % - behav task factorisation 
    load([dir_behav_scan 'taus_3Dvs2D_scan_granada.mat'])
    taus = taus.blocked.both(:,1);
    taus(badsubs_total) = [];
    X(7,:) = taus;
    labels{7} = 'fBHV';

    % - neural task factorisation BA39    
    load([dir_rsa 'groupAvg_modelBetas_cval_slRSA_paper__' rsa_rois{1} '.mat']);
    c = results.corrs(:,3);
    c(badsubs_behav) = [];
    X(8,:) = c;
    labels{8} = 'fPAR';
    % - neural task factorisation DLPFC
    load([dir_rsa 'groupAvg_modelBetas_cval_slRSA_paper__' rsa_rois{2} '.mat']);
    c = results.corrs(:,3);
    c(badsubs_behav) = [];
    X(9,:) = c;
    labels{9} = 'fRIFG';
    % - neural compression BA39
    load([dir_rsa 'results_compressangle_' rsa_rois{1} '.mat']);
    xy = [];
    for ii = 1:size(results,1)
        [~,m] = max(squeeze(results(ii,:,:)),[],'all','linear');
        [xy(ii,2),xy(ii,1)] = ind2sub([size(results,2),size(results,3)],m);
    end
    c = xy(:,2)/101;
    c(badsubs_behav) = [];  
    X(10,:) = c;
    labels{10} ='cPAR';
    % - neural compression DLPFC
    load([dir_rsa 'results_compressangle_' rsa_rois{2} '.mat']);
    xy = [];
    for ii = 1:size(results,1)
        [~,m] = max(squeeze(results(ii,:,:)),[],'all','linear');
        [xy(ii,2),xy(ii,1)] = ind2sub([size(results,2),size(results,3)],m);
    end
    c = xy(:,2)/101;
    c(badsubs_behav) = [];  
    X(11,:) = c;
    labels{11} = 'cRIFG';

    % obs x conds
    X = X';
    % transform due to normality violations
    for ii=1:size(X,2)
        [h,p] = lillietest(X(:,ii));
        if h==1
            disp('transform') 
            X(:,ii) = log(X(:,ii)+1-min(X(:,ii)));
        end
    end
    % normalise 
    X = zscore(X,1);
    N = size(X,1);
   
    %% CORRELATIONS
    
    

    % pearson
    figure();set(gcf,'Color','w');
    corrplot(X,'varnames',labels,'testR','on'); set(gcf,'Color','w');
    title('Correlation Matrix, Pearson''s \it{r}');
    %kendall
    figure();set(gcf,'Color','w');
    corrplot(X,'varnames',labels,'testR','on','type','Spearman'); set(gcf,'Color','w');
    % title('Correlation Matrix, Kendall''s \tau_{a}')
    title('Correlation Matrix, Spearman''s \rho')

    %% perform PCA 
    [c,s,l] = pca(X);
    [U,S,V] = svd(X); % to assess equivalence

    %% - Sree
    figure(); set(gcf,'Color','w');
    evs = diag(S).^2./(N-1);
    evar = evs./sum(evs);
    % plot(1:length(evar),evar,'k','LineWidth',1,'LineStyle','--');
    hold on;
    scatter(1:length(evar),evar,'Marker','diamond','MarkerFaceColor',[0,0,0],'MarkerEdgeColor',[.2 .2 .2])
    xlabel('principal component');
    ylabel('variance explained (%)');
    ylim([0,1])
    box off
    set(gca,'XTick',1:12)
    xlim([0,11]);
    set(gca,'YTickLabel',get(gca,'YTick')*100);
    title('Eigenvalue Spectrum');
    
    %% - Biplots
    cols = [1,0,0;1,0,0;1,0,0;1,.4,0;1,.4,0;0,0,0;0,.5,1;0,0.2,1;0,0.2,1;0,.5,0;0,.5,0]
    markers ={'square','square','square','diamond','diamond','diamond','diamond','square','square','square','square'};
    figure(); set(gcf,'Color','w');
    % subplot(1,3,1);
    b1 = biplot(c(:,1:2),'Scores',s(:,1:2),'VarLabels',labels);
    grid off;
    helper_change_layout(b1,cols,markers,N);
    title('PCA Projection and Loadings');
    % subplot(1,3,2);
    % b2 = biplot(c(:,2:3),'Scores',s(:,2:3),'VarLabels',labels);
    % xlabel('Component 2');
    % ylabel('Component 3');
    % grid off;
    % helper_change_layout(b2,cols,markers,N);
    % subplot(1,3,3);
    % b3 = biplot(c(:,[1,3]),'Scores',s(:,[1,3]),'VarLabels',labels);
    % xlabel('Component 1');
    % ylabel('Component 3');
    % grid off;
    % helper_change_layout(b3,cols,markers,N);


end

function helper_change_layout(b,cols,markers,N)

    for ii =1:length(markers)
        % change ...
        % color of vector 
        b(ii).Color = cols(ii,:);
        b(ii).LineWidth = 1;
        % color of marker 
        b(ii+length(markers)).MarkerFaceColor = cols(ii,:);
        b(ii+length(markers)).MarkerEdgeColor = 'None';
        % color of text
        b(ii+2*length(markers)).Color = cols(ii,:);
        % appearance of loadings
        b(ii+length(markers)).Marker = markers{ii};
        b(ii+length(markers)).MarkerSize = 5;
    end
    %% change appearance of scatter 
    for ii = 1:N 
        b(ii+3*length(markers)).Marker = 'o';
        b(ii+3*length(markers)).MarkerSize = 3;
        b(ii+3*length(markers)).MarkerFaceColor = 'None';
        b(ii+3*length(markers)).MarkerEdgeColor = [.1 .1 .1];
        
    end
    
end