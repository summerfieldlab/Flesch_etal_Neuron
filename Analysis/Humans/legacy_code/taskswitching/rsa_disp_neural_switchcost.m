function rsa_disp_neural_switchcost()
    %% rsa_disp_neural_switchcost()
    %
    % displays difference in model correlations between switch and stay trials
    % for cross-validated ROIs
    % 
    % Timo Flesch, 2020

    % rois = {'leftEVC_mod1_','rightEVC_mod1_','rightPAR_mod3_','rightIFG_mod3_'};
    % roi_labels = {'left EVC', 'right EVC', 'right Parietal', 'right DLPFC'};
    rois = {'leftifg_switch'};
    roi_labels = {'left DLPFC, no xval'};
    data_dir = 'project/results_paper/GLM_7_RSA_SWITCHSTAY/GROUP/tril/';
    models = {'grid','rotated grid','orthogonal','parallel'};
    for roi_id = 1:length(rois)
        % load beta weights for switch and stay trials 
        disp([data_dir, 'groupAvg_modelBetas_glm7_switch__', rois{roi_id}])
        load([data_dir, 'groupAvg_modelBetas_glm7_switch__', rois{roi_id}]);
        corrs_switch = results.corrs;
        load([data_dir, 'groupAvg_modelBetas_glm7_stay__', rois{roi_id}]);
        corrs_stay = results.corrs;
        % put em all in one matrix
        corrs = transpose([mean(corrs_stay,1);mean(corrs_switch,1)]);
        corrs_err = transpose([std(corrs_stay,0,1);std(corrs_switch,0,1)]./sqrt(31));
        figure(roi_id);set(gcf,'Color','w');
        b = barwitherr(corrs_err,corrs);
        hold on;
        set(gca,'XTickLabels',models);
        xlabel('Model RDM');
        set(get(gca, 'YLabel'), 'String', 'parameter estimate (a.u.)');
        for ii = 1:4
            [~,pvals(ii)] = ttest(corrs_stay(:,ii),corrs_switch(:,ii));
        end
        sigstar({[0.9,1.1],[1.9,2.1],[2.9,3.1],[3.9,4.1]},pvals);
        title(['\rm Model RSA, Switch vs Stay trials, ' roi_labels{roi_id}]);
        lgd = legend([b(1),b(2)],{'stay','switch'})
        set(lgd,'Box','off');
        
        box off
    end 
