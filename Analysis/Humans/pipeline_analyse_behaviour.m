function pipeline_analyse_behaviour(repoDir)
    %% PIPELINE_ANALYSE_BEHAVIOUR(REPODIR)
    %
    % performs statistical analyses of behavioural data

    
    results = struct();
    sem = @(X, dim) std(X, 0, dim) ./ sqrt(size(X, dim));

    badsubs = [19, 28]; % participants to exclude as no training data available (performed at 81 and 64% in scanner)


    %% 0. import data
    disp('loading data...');
    allData_train = load_struct([repoDir 'Data/Humans/part2_onlinetraining/'], 'allData_training.mat');
    allData_refresher = load_struct([repoDir 'Data/Humans/part3_fmri/behav/refresher/'], 'allData_refresher.mat');
    allData_scan = load_struct([repoDir 'Data/Humans/part3_fmri/behav/scan/'], 'allData_scan.mat');
    rsData_train = load_struct([repoDir 'Data/Humans/part2_onlinetraining/'], 'rsData_training.mat');
    rsData_refresher = load_struct([repoDir 'Data/Humans/part3_fmri/behav/refresher/'], 'rsData_refresher.mat');
    rsData_scan = load_struct([repoDir 'Data/Humans/part3_fmri/behav/scan/'], 'rsData_scan.mat');    
    disp('...done');
    
    %% 1. Accuracy: calculate training and test acc
    disp('computing average accuracy and performing statistical tests...');
    % calculate mean accuracies:
    acc_training = compute_accuracy_training(allData_train);
    acc_test = compute_accuracy_test(allData_scan);
    results.acc_baseline = acc_training.complete.p2;
    results.acc_scan = acc_test.complete;
    % perform statistical inference:
    acc_train = acc_training.complete.p2';
    acc_scan = acc_test.complete';
    acc_scan(badsubs) =[];
    [~, pval, ~, s] = ttest(acc_train, acc_scan);
    d = compute_cohensD('t', mean(acc_train, 1), std(acc_train, 0, 1), mean(acc_scan, 1), std(acc_scan, 0, 1));
    disp('accuracy, group means:')
    disp(['mu baseline ' num2str(mean(acc_train, 1)) ', sem ' num2str(sem(acc_train, 1)) ' mu scan ' num2str(mean(acc_scan, 1)) ' sem ' num2str(sem(acc_scan, 1))]);
    disp(['acc_baseline vs acc_scan, t(' num2str(s.df) ') = ' num2str(round(s.tstat, 3)) ', p = ' num2str(round(pval, 4)) ', d = ' num2str(round(d, 3))]);      

    acc_train_switch = acc_training.taskSwitchStay.p2.switch;
    acc_train_stay = acc_training.taskSwitchStay.p2.stay;    
    acc_scan_switch = acc_test.taskSwitchStay.switch;
    acc_scan_stay = acc_test.taskSwitchStay.stay;
    acc_scan_switch(badsubs) = [];
    acc_scan_stay(badsubs) = [];    
    disp('accuracy, task switch vs stay:')
    disp(['mu baseline diff ' num2str(mean(acc_train_stay - acc_train_switch, 1)) ', sem ' num2str(sem(acc_train_stay - acc_train_switch, 1)) ' mu scan diff ' num2str(mean(acc_scan_stay - acc_scan_switch, 1)) ' sem ' num2str(sem(acc_scan_stay - acc_scan_switch, 1))]);    
    [~, pval, ~, s] = ttest(acc_train_stay, acc_train_switch);
    d = compute_cohensD('t', mean(acc_train_stay, 1), std(acc_train_stay, 0, 1), mean(acc_train_switch, 1), std(acc_train_switch, 0, 1));
    
    disp(['baseline stay vs switch, t(' num2str(s.df) ') = ' num2str(round(s.tstat, 3)) ', p = ' num2str(round(pval, 4)) ', d = ' num2str(round(d, 3))]);    
    
    [~, pval, ~, s] = ttest(acc_scan_stay, acc_scan_switch);
    d = compute_cohensD('t', mean(acc_scan_stay, 1), std(acc_scan_stay, 0, 1), mean(acc_scan_switch, 1), std(acc_scan_switch, 0, 1));
    disp(['scan stay vs switch, t(' num2str(s.df) ') = ' num2str(round(s.tstat, 3)) ', p = ' num2str(round(pval, 4)) ', d = ' num2str(round(d, 3))]);
    
    [~, pval, ~, s] = ttest(acc_scan_stay - acc_scan_switch, acc_train_stay - acc_train_switch);
    d = compute_cohensD('t', mean(acc_scan_stay - acc_scan_switch, 1), std(acc_scan_stay - acc_scan_switch, 0, 1), mean(acc_train_stay - acc_train_switch, 1), std(acc_train_stay - acc_train_switch, 0, 1));
    disp(['diff of diffs, t(' num2str(s.df) ') = ' num2str(round(s.tstat, 3)) ', p = ' num2str(round(pval, 4)) ', d = ' num2str(round(d, 3))]);    
    disp('...done');
    fprintf('\n');
    fprintf('\n');
    
    

    %% 2. Choice Matrices: calculate choice matrices
    disp('computing choice matrices and sigmoids and performing statistical tests...');
    choicepatterns_training = compute_choicepatterns_training(rsData_train);
    choicepatterns_test = compute_choicepatterns_test(rsData_scan);
    results.cmats_baseline = choicepatterns_training.p2.choicemat.aligned;
    results.cmats_scan = choicepatterns_test.choicemat.aligned;
    % fit 1 param sigmoids: 
    sigmoidfits_baseline = regress_choiceSigmoids(choicepatterns_training.p2);
    sigmoidfits_scan = regress_choiceSigmoids(choicepatterns_test);
    results.sigmoids_baseline = sigmoidfits_baseline.sigmas;
    results.sigmoids_scan = sigmoidfits_scan.sigmas;
    % statistical inference:
    sigmas_train_irrel = results.sigmoids_baseline.irrel;
    sigmas_train_rel = results.sigmoids_baseline.rel;    
    sigmas_scan_irrel = results.sigmoids_scan.irrel;
    sigmas_scan_rel = results.sigmoids_scan.rel;
    sigmas_scan_irrel(badsubs) = [];
    sigmas_scan_rel(badsubs) = [];

    % irrel: each against zero
    disp(' --- irrelevant dimension -----')
    % .. baseline
    [pval, ~, s] = signrank(sigmas_train_irrel);
    d = compute_nonparametricEffectsize(s.zval, length(sigmas_train_irrel));
    disp(['baseline irrel, z = ' num2str(round(s.zval, 3)) ', p = ' num2str(round(pval, 4)) ', d = ' num2str(round(d, 3))]);
    % .. scan
    [pval, ~, s] = signrank(sigmas_scan_irrel);
    d = compute_nonparametricEffectsize(s.zval, length(sigmas_scan_irrel));
    disp(['scan irrel, z = ' num2str(round(s.zval, 3)) ', p = ' num2str(round(pval, 4)) ', d = ' num2str(round(d, 3))]);

    % irrel: baseline vs scan
    [pval, ~, s] = signrank(sigmas_scan_irrel, sigmas_train_irrel);
    d = compute_nonparametricEffectsize(s.zval, length(sigmas_train_irrel));
    disp(['scan minus baseline irrel, z = ' num2str(round(s.zval, 3)) ', p = ' num2str(round(pval, 4)) ', d = ' num2str(round(d, 3))]);

    fprintf('\n')
    % irrel: each against zero
    disp(' --- relevant dimension -----')
    % .. baseline
    [pval, ~, s] = signrank(sigmas_train_rel);
    d = compute_nonparametricEffectsize(s.zval, length(sigmas_train_rel));
    disp(['baseline rel, z = ' num2str(round(s.zval, 3)) ', p = ' num2str(round(pval, 4)) ', d = ' num2str(round(d, 3))]);
    % .. scan
    [pval, ~, s] = signrank(sigmas_scan_rel);
    d = compute_nonparametricEffectsize(s.zval, length(sigmas_scan_rel));
    disp(['scan rel, z = ' num2str(round(s.zval, 3)) ', p = ' num2str(round(pval, 4)) ', d = ' num2str(round(d, 3))]);

    % rel: baseline vs scan
    [pval, ~, s] = signrank(sigmas_scan_rel, sigmas_train_rel);
    d = compute_nonparametricEffectsize(s.zval, length(sigmas_train_rel));
    disp(['scan minus baseline rel, z = ' num2str(round(s.zval, 3)) ', p = ' num2str(round(pval, 4)) ', d = ' num2str(round(d, 3))]);

    fprintf('\n')
    % irrel: each against zero
    disp(' --- relevant vs irrelevant dimension -----')
    [pval, ~, s] = signrank(sigmas_scan_rel, sigmas_scan_irrel);
    d = compute_nonparametricEffectsize(s.zval, length(sigmas_scan_irrel));
    disp(['scan rel minus irrel, z = ' num2str(round(s.zval, 3)) ', p = ' num2str(round(pval, 4)) ', d = ' num2str(round(d, 3))]);
    [pval, ~, s] = signrank(sigmas_train_rel, sigmas_train_irrel);
    d = compute_nonparametricEffectsize(s.zval, length(sigmas_train_irrel));
    disp(['train rel minus irrel, z = ' num2str(round(s.zval, 3)) ', p = ' num2str(round(pval, 4)) ', d = ' num2str(round(d, 3))]);
    fprintf('\n');
    [pval, ~, s] = signrank(sigmas_scan_rel - sigmas_scan_irrel, sigmas_train_rel - sigmas_train_irrel);
    d = compute_nonparametricEffectsize(s.zval, length(sigmas_train_irrel));
    disp(['train vs test rel minus irrel, z = ' num2str(round(s.zval, 3)) ', p = ' num2str(round(pval, 4)) ', d = ' num2str(round(d, 3))]);

    disp('...done');
    
    %% 3. Choice RSA: fit factorised/linear models
    % compute rdms
    disp('computing behavioural rdms...');
    cmats_baseline = choicepatterns_training.p2.choicemat;
    cmats_scan = choicepatterns_test.choicemat;
    rdms_baseline = rsa_computeBehavRDMs(cmats_baseline);
    rdms_scan = rsa_computeBehavRDMs(cmats_scan);
    results.rdms_baseline = rdms_baseline.blocked.both;
    results.rdms_scan = rdms_scan.blocked.both;
    disp('...done');
    
    % regress against model rdms:
    disp('performing behavioural RSA...');
    betas_baseline = rsa_regressRDMs(rdms_baseline);
    betas_scan = rsa_regressRDMs(rdms_scan);
    results.behavrsa_baseline = betas_baseline.blocked.both;
    results.behavrsa_scan = betas_scan.blocked.both;
    % statistical inference:
    results.behavrsa_scan(badsubs,:) = [];
    % factorised vs lin: baseline, scan
    [~, pval, ~, s] = ttest(results.behavrsa_baseline(:,1), results.behavrsa_baseline(:,2));
    d = compute_cohensD('t', mean(results.behavrsa_baseline(:,1), 1), std(results.behavrsa_baseline(:,1), 0, 1), mean(results.behavrsa_baseline(:,2), 1), std(results.behavrsa_baseline(:,2), 0, 1));
    disp(['baseline factorised vs linear model, t(' num2str(s.df) ') = ' num2str(round(s.tstat, 3)) ', p = ' num2str(round(pval, 4)) ', d = ' num2str(round(d, 3))]);

    [~, pval, ~, s] = ttest(results.behavrsa_scan(:,1), results.behavrsa_scan(:,2));
    d = compute_cohensD('t', mean(results.behavrsa_scan(:,1), 1), std(results.behavrsa_scan(:,1), 0, 1), mean(results.behavrsa_scan(:,2), 1), std(results.behavrsa_scan(:,2), 0, 1));
    disp(['scan factorised vs linear model, t(' num2str(s.df) ') = ' num2str(round(s.tstat, 3)) ', p = ' num2str(round(pval, 4)) ', d = ' num2str(round(d, 3))]);

    % baseline vs scan: factorised-lin
    [~, pval, ~, s] = ttest(results.behavrsa_baseline(:,1)-results.behavrsa_baseline(:,2), results.behavrsa_scan(:,1)-results.behavrsa_scan(:,2));
    d = compute_cohensD('t', mean(results.behavrsa_baseline(:,1)-results.behavrsa_baseline(:,2), 1), std(results.behavrsa_baseline(:,1)-results.behavrsa_baseline(:,2), 0, 1), mean(results.behavrsa_scan(:,1)-results.behavrsa_scan(:,2), 1), std(results.behavrsa_scan(:,1)-results.behavrsa_scan(:,2), 0, 1));
    disp(['baseline factorised vs linear model, t(' num2str(s.df) ') = ' num2str(round(s.tstat, 3)) ', p = ' num2str(round(pval, 4)) ', d = ' num2str(round(d, 3))]);
    disp('...done');

    %% 4. Psychophysical model: fit psychophysical model
    disp('fitting psychophysical model...');
    modelfits_baseline = compute_choicemodel_aligned(cmats_baseline.aligned);
    modelfits_scan = compute_choicemodel_aligned(cmats_scan.aligned);
    results.modelfits_baseline = modelfits_baseline;
    results.modelfits_scan = modelfits_scan;
    % statistical inference:    
    fns = fieldnames(modelfits_scan);
    for ii = 1:length(fns)
        modelfits_scan.(fns{ii})(badsubs) = [];
    end

    disp('--- bias -----')
    % bias: each against zero
    [pval, ~, s] = signrank(abs(modelfits_baseline.bias));
    d = compute_nonparametricEffectsize(s.zval, length(modelfits_baseline.bias));
    disp(['baseline bias, z = ' num2str(round(s.zval, 3)) ', p = ' num2str(round(pval, 4)) ', d = ' num2str(round(d, 3))]);

    [pval, ~, s] = signrank(abs(modelfits_scan.bias));
    d = compute_nonparametricEffectsize(s.zval, length(modelfits_scan.bias));
    disp(['scan bias, z = ' num2str(round(s.zval, 3)) ', p = ' num2str(round(pval, 4)) ', d = ' num2str(round(d, 3))]);
    fprintf('\n');
    % abs bias: baseline vs scan
    [pval, ~, s] = signrank(abs(modelfits_scan.bias) - abs(modelfits_baseline.bias));
    d = compute_nonparametricEffectsize(s.zval, length(modelfits_scan.bias));
    disp(['scan minus baseline abs bias, z = ' num2str(round(s.zval, 3)) ', p = ' num2str(round(pval, 4)) ', d = ' num2str(round(d, 3))]);
    fprintf('\n');
    disp('--- lapse ----')
    % lapses: baseline vs scan
    [pval, ~, s] = signrank(modelfits_scan.lapse - modelfits_baseline.lapse);
    d = compute_nonparametricEffectsize(s.zval, length(modelfits_scan.lapse));
    disp(['scan minus baseline lapse, z = ' num2str(round(s.zval, 3)) ', p = ' num2str(round(pval, 4)) ', d = ' num2str(round(d, 3))]);
    fprintf('\n');
    disp('--- slope ----')
    % slope: baseline vs scan
    [pval, ~, s] = signrank(modelfits_scan.slope - modelfits_baseline.slope);
    d = compute_nonparametricEffectsize(s.zval, length(modelfits_scan.slope));
    disp(['scan minus baseline slope, z = ' num2str(round(s.zval, 3)) ', p = ' num2str(round(pval, 4)) ', d = ' num2str(round(d, 3))]);
    fprintf('\n');
    disp('--- offset ----')
    % offset: baseline vs scan
    [pval, ~, s] = signrank(modelfits_scan.offset - modelfits_baseline.offset);
    d = compute_nonparametricEffectsize(s.zval, length(modelfits_scan.offset));
    disp(['scan minus baseline offset, z = ' num2str(round(s.zval, 3)) ', p = ' num2str(round(pval, 4)) ', d = ' num2str(round(d, 3))]);
    disp('...done');

    %% 5. export results for plotting
    disp('saving results...');
    save('results_behaviour.mat', 'results')
    disp('...done');

    %% TODO plot some low quality figures for si
    % - switch vs stay acc: import from disp_paper_accuracy()
    % - sigmoids. import from disp_paper_choice_sigmoids()

end

function acc_all = compute_accuracy_training(goodData)
    %% init data structs
    acc_all = struct();

    % learning curves
    acc = helper_compute_lcurves_training(goodData);

    %% compute overall accuracy
    acc_all.complete.p1 = nanmean(acc.pMat(goodData.expt_phase(:, 1) == 1, :), 1); % accuracy first training
    acc_all.complete.p2 = nanmean(acc.pMat(goodData.expt_phase(:, 1) == 2, :), 1); % accuracy first test
    acc_all.complete.p3 = nanmean(acc.pMat(goodData.expt_phase(:, 1) == 3, :), 1); % accuracy main training

    %% compute task switch vs task stay accuracy
    acc_all.taskSwitchStay = helper_compute_taskSwitchStayAcc_training(goodData, acc.pMat);

    %% compute key switch vs task stay accuracy
    acc_all.keySwitchStay = helper_compute_keySwitchStayAcc_training(goodData, acc.pMat);

    %% compute cat switch vs task stay accuracy
    acc_all.catSwitchStay = helper_compute_catSwitchStayAcc_training(goodData, acc.pMat);

    function acc_all = helper_compute_taskSwitchStayAcc_training(goodData, pMat)
        %
        % computes accuracy separately for task switch and task stay trials
        phaseNames = {'p1', 'p2', 'p3'};

        for subj = 1:size(goodData.subCodes, 2)

            for p = 1:length(phaseNames)
                accVect = pMat(goodData.expt_phase(:, subj) == p, subj);
                ctxVect = goodData.expt_contextIDX(goodData.expt_phase(:, subj) == p, subj);

                stayVect = zeros(length(ctxVect), 1);
                stayVect(ctxVect == circshift(ctxVect, 1), 1) = 1;
                switchVect = zeros(length(ctxVect), 1);
                switchVect(~stayVect) = 1;
                acc_all.(phaseNames{p}).stay(subj, :) = nanmean(accVect(stayVect == 1));
                acc_all.(phaseNames{p}).switch(subj, :) = nanmean(accVect(switchVect == 1));
            end

        end

    end

    function acc_all = helper_compute_keySwitchStayAcc_training(goodData, pMat)
        %
        % computes accuracy separately for trials where key mapping switched or stayed
        % e.g. accept on left screen on trial n and n-1 == stay

        phaseNames = {'p1', 'p2', 'p3'};

        for subj = 1:size(goodData.subCodes, 2)

            for p = 1:length(phaseNames)
                accVect = pMat(goodData.expt_phase(:, subj) == p, subj);
                keyVect = goodData.expt_keyassignment(goodData.expt_phase(:, subj) == p, subj);

                stayVect = zeros(length(keyVect), 1);
                stayVect(keyVect == circshift(keyVect, 1), 1) = 1;
                switchVect = zeros(length(keyVect), 1);
                switchVect(~stayVect) = 1;
                acc_all.(phaseNames{p}).stay(subj, :) = nanmean(accVect(stayVect == 1));
                acc_all.(phaseNames{p}).switch(subj, :) = nanmean(accVect(switchVect == 1));
            end

        end

    end

    function acc_all = helper_compute_catSwitchStayAcc_training(goodData, pMat)
        %
        % computes accuracy separately for trials where stim category switched or stayed

        phaseNames = {'p1', 'p2', 'p3'};

        for subj = 1:size(goodData.subCodes, 2)

            for p = 1:length(phaseNames)
                accVect = pMat(goodData.expt_phase(:, subj) == p, subj);
                keyVect = goodData.expt_catIDX(goodData.expt_phase(:, subj) == p, subj);

                stayVect = zeros(length(keyVect), 1);
                stayVect(keyVect == circshift(keyVect, 1), 1) = 1;
                switchVect = zeros(length(keyVect), 1);
                switchVect(~stayVect) = 1;
                acc_all.(phaseNames{p}).stay(subj, :) = nanmean(accVect(stayVect == 1));
                acc_all.(phaseNames{p}).switch(subj, :) = nanmean(accVect(switchVect == 1));
            end

        end

    end

    function results = helper_compute_lcurves_training(allData)
        results = struct();

        numSubs = length(allData.subSubject);

        for subj = 1:numSubs
            catZero = find(allData.expt_catIDX(:, subj) == 0);
            corrAll = allData.resp_correct(:, subj);
            corrCat = corrAll;
            timeOuts = find(allData.resp_reactiontime(:, subj) > 3);

            corrCat(timeOuts) = 0;
            corrCat(catZero) = NaN;
            pMat(:, subj) = corrCat;

        end

        results.pMat = pMat;
    end

end

function acc_all = compute_accuracy_test(goodData)
    %% ACC_ALL = compute_accuracy_test(GOODDATA)
    %
    % computes mean accuracy for training and test, and for switch vs stay at test

    %% init data structs
    acc_all = struct();

    % learning curves
    acc = helper_compute_lcurves_test(goodData);

    %% compute overall accuracy
    acc_all.complete = nanmean(acc.pMat, 1); % accuracy first training

    %% compute task switch vs task stay accuracy
    acc_all.taskSwitchStay = helper_compute_taskSwitchStayAcc_test(goodData, acc.pMat);

    %% compute key switch vs task stay accuracy
    acc_all.keySwitchStay = helper_compute_keySwitchStayAcc_test(goodData, acc.pMat);

    %% compute cat switch vs task stay accuracy
    acc_all.catSwitchStay = helper_compute_catSwitchStayAcc_test(goodData, acc.pMat);

    %% compute first vs second task accuracy
    acc_all.taskFirstVsSecond = helper_compute_firstVsSecondAcc_test(goodData, acc.pMat);

    function results = helper_compute_lcurves_test(allData)
        %% helper_compute_lcurves_test(ALLDATA)
        %
        % computes learning curves in terms of correct classifications
        % sets the category boundary trials to NaN
        % Timo Flesch, 2018

        results = struct();

        numSubs = length(allData.ruleID);

        for subj = 1:numSubs
            catZero = find(allData.expt_catIDX(subj, :) == 0);
            corrAll = allData.resp_correct(subj, :);
            corrCat = corrAll;
            timeOuts = find(allData.resp_reactiontime(subj, :) > 3);
            corrCat = cast(corrCat, 'double');
            corrCat(timeOuts) = 0;
            corrCat(catZero) = NaN;
            pMat(:, subj) = corrCat;

        end

        results.pMat = pMat;

    end

    function acc_all = helper_compute_taskSwitchStayAcc_test(goodData, pMat)
        %
        % computes accuracy separately for task switch and task stay trials

        for subj = 1:size(goodData.order, 1)
            accVect = pMat(:, subj);
            ctxVect = goodData.expt_contextIDX(subj, :);
            stayVect = zeros(length(ctxVect), 1);
            stayVect(ctxVect == circshift(ctxVect, 1), 1) = 1;
            switchVect = zeros(length(ctxVect), 1);
            switchVect(~stayVect) = 1;
            acc_all.stay(subj, :) = nanmean(accVect(stayVect == 1));
            acc_all.switch(subj, :) = nanmean(accVect(switchVect == 1));
        end

    end

    function acc_all = helper_compute_keySwitchStayAcc_test(goodData, pMat)
        %
        % computes accuracy separately for trials where key mapping switched or stayed
        % e.g. accept on left screen on trial n and n-1 == stay

        for subj = 1:size(goodData.order, 1)
            accVect = pMat(:, subj);
            keyVect = goodData.keymapping(subj, :, 1);

            stayVect = zeros(length(keyVect), 1);
            stayVect(keyVect == circshift(keyVect, 1), 1) = 1;
            switchVect = zeros(length(keyVect), 1);
            switchVect(~stayVect) = 1;
            acc_all.stay(subj, :) = nanmean(accVect(stayVect == 1));
            acc_all.switch(subj, :) = nanmean(accVect(switchVect == 1));
        end

    end

    function acc_all = helper_compute_catSwitchStayAcc_test(goodData, pMat)
        %
        % computes accuracy separately for trials where stim category switched or stayed

        for subj = 1:size(goodData.order, 1)
            accVect = pMat(:, subj);
            keyVect = goodData.expt_catIDX(subj, :);

            stayVect = zeros(length(keyVect), 1);
            stayVect(keyVect == circshift(keyVect, 1), 1) = 1;
            switchVect = zeros(length(keyVect), 1);
            switchVect(~stayVect) = 1;
            acc_all.stay(subj, :) = nanmean(accVect(stayVect == 1));
            acc_all.switch(subj, :) = nanmean(accVect(switchVect == 1));
        end

    end

    function acc_all = helper_compute_firstVsSecondAcc_test(goodData, pMat)
        %
        % computes accuracy separately for first and second task the subs had been
        %  trained on the day before.

        for subj = 1:size(goodData.order, 1)
            accVect = pMat(:, subj);
            ctxVect = goodData.expt_contextIDX(subj, :);

            switch goodData.order(subj)
                case 1
                    idFirst = 1;
                    idSecond = 2;
                case 2
                    idFirst = 2;
                    idSecond = 1;
            end

            acc_all.first(subj, :) = nanmean(accVect(ctxVect == idFirst));
            acc_all.second(subj, :) = nanmean(accVect(ctxVect == idSecond));
        end

    end

end

function results = compute_choicepatterns_training(rsData)
    %% RESULTS = compute_choicepatterns_training(RSDATA)
    %
    % computes RT and choice fractions  as functions
    % of feature values, either along single dimensions or along both (e.g. as position in 2d grid)
    %
    % Timo Flesch, 2018

    rewVals = [-50, -25, 0, 25, 50];
    results = struct();
    expPhases = {'p1', 'p2', 'p3'};

    for subj = 1:length(rsData)
        % SINGLE DIMENSIONS
        for phaseIDX = 1:length(expPhases)
            subData = rsData(subj).data(rsData(subj).data(:, 2) == phaseIDX, :); % sub-select phase-specific trials

            for rewardID = 1:length(rewVals)
                thisRew = rewVals(rewardID);
                % RT as function of reward along relevant dimension
                results.(expPhases{phaseIDX}).rel.rt(subj, rewardID) = nanmean(subData(subData(:, 7) == thisRew, 11));
                % Choice as function of reward along relevant dimension
                results.(expPhases{phaseIDX}).rel.choice(subj, rewardID) = nanmean(subData(subData(:, 7) == thisRew, 8));
                % RT as function of reward along irrelevant dimension
                results.(expPhases{phaseIDX}).irrel.rt(subj, rewardID) = nanmean(subData(subData(:, 14) == thisRew, 11));
                % Choice as function of reward along irrelevant dimension
                results.(expPhases{phaseIDX}).irrel.choice(subj, rewardID) = nanmean(subData(subData(:, 14) == thisRew, 8));
            end

            % BOTH DIMENSIONS
            tmp_train_north = zeros(length(rewVals), length(rewVals));
            tmp_train_south = zeros(length(rewVals), length(rewVals));

            for leafID = 1:length(rewVals)

                for branchID = 1:length(rewVals)
                    tmp_train_north(leafID, branchID) = squeeze(nanmean(subData(subData(:, 3) == 1 & subData(:, 4) == leafID & subData(:, 5) == branchID, 8)));
                    tmp_train_south(leafID, branchID) = squeeze(nanmean(subData(subData(:, 3) == 2 & subData(:, 4) == leafID & subData(:, 5) == branchID, 8)));
                end

            end

            results.(expPhases{phaseIDX}).choicemat.orig.north(subj, :, :) = tmp_train_north;
            results.(expPhases{phaseIDX}).choicemat.orig.south(subj, :, :) = tmp_train_south;
            % bring all mats in same reference frame
            results.(expPhases{phaseIDX}).choicemat.aligned.north(subj, :, :) = helper_alignMatrices(tmp_train_north, rsData(subj).code(end), 'north');
            results.(expPhases{phaseIDX}).choicemat.aligned.south(subj, :, :) = helper_alignMatrices(tmp_train_south, rsData(subj).code(end), 'south');
            results.(expPhases{phaseIDX}).code(subj) = rsData(subj).code(end);

            % single trials
            results.(expPhases{phaseIDX}).singletrials.north.data(subj, :, :) = subData(subData(:, 3) == 1, [4, 5, 8, 12, 14]);
            results.(expPhases{phaseIDX}).singletrials.north.bound(subj) = helper_setOptimBounds(results.(expPhases{phaseIDX}).code(subj), 'north');
            results.(expPhases{phaseIDX}).singletrials.north.diag(subj) = helper_setOptimBounds(results.(expPhases{phaseIDX}).code(subj), 'diag');
            results.(expPhases{phaseIDX}).singletrials.south.data(subj, :, :) = subData(subData(:, 3) == 2, [4, 5, 8, 12, 14]);
            results.(expPhases{phaseIDX}).singletrials.south.bound(subj) = helper_setOptimBounds(results.(expPhases{phaseIDX}).code(subj), 'south');
            results.(expPhases{phaseIDX}).singletrials.south.diag(subj) = helper_setOptimBounds(results.(expPhases{phaseIDX}).code(subj), 'diag');

        end

    end

end

function results = compute_choicepatterns_test(rsData)
    %% RESULTS = compute_choicepatterns_test(RSDATA)
    %
    % computes RT and choice fractions  as functions
    % of feature values, either along single dimensions or along both (e.g. as position in 2d grid)
    %
    % Timo Flesch, 2018

    rewVals = [-50, -25, 0, 25, 50];
    results = struct();

    for subj = 1:length(rsData)
        % SINGLE DIMENSIONS

        subData = rsData(subj).data; % sub-select phase-specific trials

        for rewardID = 1:length(rewVals)
            thisRew = rewVals(rewardID);
            % RT as function of reward along relevant dimension
            results.rel.rt(subj, rewardID) = nanmean(subData(subData(:, 7) == thisRew, 11));
            % Choice as function of reward along relevant dimension
            results.rel.choice(subj, rewardID) = nanmean(subData(subData(:, 7) == thisRew, 8));
            % RT as function of reward along irrelevant dimension
            results.irrel.rt(subj, rewardID) = nanmean(subData(subData(:, 14) == thisRew, 11));
            % Choice as function of reward along irrelevant dimension
            results.irrel.choice(subj, rewardID) = nanmean(subData(subData(:, 14) == thisRew, 8));
        end

        % BOTH DIMENSIONS
        tmp_train_north = zeros(length(rewVals), length(rewVals));
        tmp_train_south = zeros(length(rewVals), length(rewVals));

        for leafID = 1:length(rewVals)

            for branchID = 1:length(rewVals)
                tmp_train_north(leafID, branchID) = squeeze(nanmean(subData(subData(:, 3) == 1 & subData(:, 4) == leafID & subData(:, 5) == branchID, 8)));
                tmp_train_south(leafID, branchID) = squeeze(nanmean(subData(subData(:, 3) == 2 & subData(:, 4) == leafID & subData(:, 5) == branchID, 8)));
            end

        end

        results.choicemat.orig.north(subj, :, :) = tmp_train_north;
        results.choicemat.orig.south(subj, :, :) = tmp_train_south;
        % bring all mats in same reference frame
        results.choicemat.aligned.north(subj, :, :) = helper_alignMatrices(tmp_train_north, rsData(subj).code(end), 'north');
        results.choicemat.aligned.south(subj, :, :) = helper_alignMatrices(tmp_train_south, rsData(subj).code(end), 'south');
        results.code(subj) = rsData(subj).code(end);

        % single runs:
        tmp_train_north = zeros(length(unique(subData(:, 1))), length(rewVals), length(rewVals));
        tmp_train_south = zeros(length(unique(subData(:, 1))), length(rewVals), length(rewVals));

        for runID = 1:length(unique(subData(:, 1)))

            for leafID = 1:length(rewVals)

                for branchID = 1:length(rewVals)
                    tmp_train_north(runID, leafID, branchID) = squeeze(nanmean(subData(subData(:, 1) == runID & subData(:, 3) == 1 & subData(:, 4) == leafID & subData(:, 5) == branchID, 8)));
                    tmp_train_south(runID, leafID, branchID) = squeeze(nanmean(subData(subData(:, 1) == runID & subData(:, 3) == 2 & subData(:, 4) == leafID & subData(:, 5) == branchID, 8)));
                end

            end

        end

        results.choicemat.orig_singleRuns.north(subj, :, :, :) = tmp_train_north;
        results.choicemat.orig_singleRuns.south(subj, :, :, :) = tmp_train_south;

    end

end


function results = regress_choiceSigmoids(choiceProbs)

    allSigmas = struct();
    allStats = struct();

    rewVals = [-2:2];
    dimNames = {'rel', 'irrel'};

    for dd = 1:length(dimNames)

        for subj = 1:size(choiceProbs.rel.choice, 1)
            y_bl = squeeze(choiceProbs.(dimNames{dd}).choice(subj, :));
            % disp('noop')
            % fit sigmoids
            sigmas = fitSigmoid([rewVals', y_bl']);
            allSigmas.(dimNames{dd})(subj, :) = sigmas;
        end

        % perform within-group test for statistical significance
        for bb = 1:size(allSigmas.(dimNames{dd}), 2)
            [p, ~, s] = signrank(squeeze(allSigmas.(dimNames{dd})(:, bb)), 0, 'method', 'approximate');
            allStats.(dimNames{dd}).p(bb) = p;
            allStats.(dimNames{dd}).z(bb) = s.zval;
        end

    end

    results.sigmas = allSigmas;
    results.stats = allStats;

    function betas = fitSigmoid(data)
        betas = nlinfit(data(:, 1), data(:, 2), @mysigmoid, [.1]);
    end

    function fun = mysigmoid(b, x)
        fun = 1 ./ (1 + exp(-b(1) * (x)));
    end

end


function optimBound = helper_setOptimBounds(code, garden)
    %% HELPER_SETOPTIMBOUNDS(CODE,GARDEN)
    %
    % sets optimal boundary for each task
    %
    % inputs:
    % - code: reward assignment (1-4)
    % - garden: context (north, south)
    %
    % Timo Flesch, 2018

    optimBound = 0;

    switch code
        case 1

            if strcmp(garden, 'north')
                optimBound = 180;
            elseif strcmp(garden, 'south')
                optimBound = 90;
            elseif strcmp(garden, 'diag')
                optimBound = 135;
            end

        case 2

            if strcmp(garden, 'north')
                optimBound = 0;
            elseif strcmp(garden, 'south')
                optimBound = 270;
            elseif strcmp(garden, 'diag')
                optimBound = 315;
            end

        case 3

            if strcmp(garden, 'north')
                optimBound = 0;
            elseif strcmp(garden, 'south')
                optimBound = 90;
            elseif strcmp(garden, 'diag')
                optimBound = 45;
            end

        case 4

            if strcmp(garden, 'north')
                optimBound = 180;
            elseif strcmp(garden, 'south')
                optimBound = 270;
            elseif strcmp(garden, 'diag')
                optimBound = 225;
            end

    end

end

function outMat = helper_alignMatrices(inMat, code, garden)
    %% HELPER_ALIGNMATRICES(INMAT,CODE,GARDEN)
    %
    % aligns all matrices of trees task to have same frame of reference
    %
    % input:
    %	- inMat:   matrix to manipulate
    %	- code:    code of reward assignment schema (1 to 8)
    % - garden: 'north' or 'south'
    %
    % Timo Flesch, 2018

    switch (code)
        case 1
            % cardinal high high
            if strcmp(garden, 'north')
                outMat = inMat;
            elseif strcmp(garden, 'south')
                outMat = inMat;
            end

        case 2
            % cardinal low low
            if strcmp(garden, 'north')
                outMat = flipud(fliplr(inMat));
            elseif strcmp(garden, 'south')
                outMat = flipud(fliplr(inMat));
            end

        case 3
            % cardinal low high
            if strcmp(garden, 'north')
                outMat = flipud(inMat);
            elseif strcmp(garden, 'south')
                outMat = flipud(inMat);
            end

        case 4
            % cardinal high low
            if strcmp(garden, 'north')
                outMat = fliplr(inMat);
            elseif strcmp(garden, 'south')
                outMat = fliplr(inMat);
            end

    end

end

function rdmCollection = rsa_computeBehavRDMs(results)

    rdmCollection = struct();
    dimIn = {'north', 'south'};
    dimOut = {'le', 'br'};

    % all subs
    for subID = 1:(size(results.aligned.north, 1))
        % north/south -> le/br
        for dimID = 1:length(dimIn)
            % blocked
            tmp = squeeze(results.aligned.(dimIn{dimID})(subID, :, :));
            rdmCollection.blocked.(dimOut{dimID})(subID, :, :) = squareform(pdist(tmp(:)));
        end

        % blocked - both (north/branch, south/leaf)
        matNorth = squeeze(results.aligned.north(subID, :, :));
        matSouth = squeeze(results.aligned.south(subID, :, :));
        tmp = [matNorth(:); matSouth(:)];
        rdmCollection.blocked.both(subID, :, :) = squareform(pdist(tmp(:)));
    end

end

function betas = rsa_regressRDMs(rdmCollection)

    betas = struct();

    groupNames = {'blocked'};

    modLeaf = rsa_genCategoricalModelRDM(1);
    modBranch = rsa_genCategoricalModelRDM(2);
    modDiag = rsa_genCategoricalModelRDM(3); % diag 1 is interaction

    rdm3Dmod = squareform(pdist([modLeaf.choiceMat(:); modBranch.choiceMat(:)]));
    rdm2Dmod = squareform(pdist([modDiag.choiceMat(:); modDiag.choiceMat(:)]));
    dmat = [];
    dmat(:, 1) = zscore(vectorizeRDM(rdm3Dmod));
    dmat(:, 2) = zscore(vectorizeRDM(rdm2Dmod));

    for grpID = 1:length(groupNames)
        % all subs
        for subID = 1:(size(rdmCollection.(groupNames{grpID}).both, 1))

            behavRDM = transpose(zscore(vectorizeRDM(squeeze(rdmCollection.(groupNames{grpID}).both(subID, :, :)))));

            b = regress(behavRDM, dmat);
            betas.(groupNames{grpID}).both(subID, 1) = b(1);
            betas.(groupNames{grpID}).both(subID, 2) = b(2);

        end

    end

    function myModel = rsa_genCategoricalModelRDM(modIDX)

        myModel = struct();

        %% main
        % 1. compute choice matrix:
        switch modIDX
            case 1
                choiceMat = repmat([0; 0; 0.5; 1; 1], 1, 5); % cardinal north
            case 2
                choiceMat = repmat([0, 0, 0.5, 1, 1], 5, 1); % cardinal south (ortho diagonal)
            case 3
                choiceMat = tril(ones(5, 5)); % diagonal north (ortho cardinal)
                choiceMat(1:length(choiceMat) + 1:end) = 0.5;
                choiceMat = fliplr(choiceMat);
            case 4
                choiceMat = tril(ones(5, 5)); % diagonal south
                choiceMat(1:length(choiceMat) + 1:end) = 0.5;
                choiceMat = rot90(choiceMat, 2);

        end

        % 2. compute rdm:
        myModel.choiceMat = choiceMat;
        myModel.choiceRDM = squareform(pdist(choiceMat(:)));
    end

end

function results = compute_choicemodel_aligned(choicemats)
    %% compute_choicemodel()
    %
    % fits parametric choice model to behavioural data
    % to estimate lapse rates, sigmoid bias/slope and
    % deviation between learned and true decision boundary
    %
    % input: results.choicemat.aligned (subfields north and south)
    % Timo Flesch, 2019,
    % Human Information Processing Lab,
    % Experimental Psychology Department
    % University of Oxford

    %% set params
    cat_bounds = [180, 90]; %optimal boundaries, factorised model
    init_vals = [cat_bounds, 0, 20, 0]; % 180 for north, 90 for south task, zero offset and lapse, binarized choice probs
    constraints = struct();
    constraints.bound_north = [90, 270];
    constraints.bound_south = [0, 180];
    constraints.offset = [-1, 1];
    constraints.slope = [0, 20];
    constraints.lapserate = [0, 0.5];
    constraints_all = [constraints.bound_north; constraints.bound_south; constraints.offset; constraints.slope; constraints.lapserate];

    %% estimate model (single subject level)
    results = struct();
    [a, b] = meshgrid(-2:2, -2:2);
    x = [a(:), b(:)];
    % fid = 1;
    % figure();set(gcf,'Color','w');
    for ii = 1:size(choicemats.north, 1)
        % disp(['Processing subject ' num2str(ii)]);
        cm_north = squeeze(choicemats.north(ii, :, :));
        cm_south = squeeze(choicemats.south(ii, :, :));
        y_true = [cm_north(:); cm_south(:)];
        [thetas, negLL] = fit_model(x, y_true, init_vals, constraints_all(:, 1), constraints_all(:, 2));

        bias = [compute_boundaryBias(thetas(1), cat_bounds(1), 'north'), ...
                compute_boundaryBias(thetas(2), cat_bounds(2), 'south')];
        % five parameter model (phi_A,phi_B,slope,offset,lapse)
        results.slope(ii, :) = thetas(4);
        results.offset(ii, :) = thetas(3);
        results.lapse(ii, :) = thetas(5);
        results.bias(ii, :) = squeeze(mean(bias));
        results.phi(ii, :) = squeeze(thetas(1:2));
        results.nll(ii, :) = negLL;
        results.bic(ii, :) = compute_BIC(-negLL, length(thetas), 50); % goodness of fit

        % y_hat = choicemodel(x,thetas);
        % y_north = reshape(y_hat(1:25),[5,5]);
        % y_south = reshape(y_hat(26:end),[5,5]);
        % subplot(8,16,fid);
        % imagesc(cm_north);
        % axis square;
        % % title(num2str(allData.ruleID(ii)));
        % subplot(8,16,fid+16);
        % imagesc(cm_south);
        % axis square;
        % subplot(8,16,fid+32);
        % imagesc(y_north);
        % subplot(8,16,fid+48);
        % imagesc(y_south);

        % if fid==16
        %     fid = 64;
        % end
        % fid = fid+1;
    end

end

function y_hat = choicemodel(X, theta)
    %
    % a parametric model for single subject choices
    X1 = scalarproj(X, theta(1));
    X2 = scalarproj(X, theta(2));
    y_hat = transducer([X1; X2], theta(3), theta(4), theta(5));

    function y = scalarproj(x, phi)
        phi_bound = deg2rad(phi);
        phi_ort = phi_bound - deg2rad(90);
        y = x * [cos(phi_ort); sin(phi_ort)];
    end

    function y = transducer(x, offset, slope, lapse)
        y = lapse + (1 - lapse * 2) ./ (1 + exp(-slope * (x - offset)));
    end

end

function [betas, loss] = fit_model(x, y_true, init_vals, lb, ub)
    %
    % minimises objective function
    % returns best fitting parameters

    % define objective function
    loss = @(init_vals) - sum(log(1 - abs(y_true(:) - choicemodel(x, init_vals)) + 1e-10));

    % fit model
    [betas, loss] = fmincon(loss, init_vals, [], [], [], [], lb, ub, [], optimoptions('fmincon', 'Display', 'off'));
end

function boundaryBias = compute_boundaryBias(estimatedAngle, catBound, task)
    % we interpret a higher positive bias as stronger tendency towards a combined representations
    % hence the sign flip for north (combined would be 90, but optimal is 180 deg)
    switch task
        case 'north'
            % boundaryBias = abs(rad2deg(circ_dist(deg2rad(estimatedAngle),deg2rad(catBound))));
            boundaryBias = -(rad2deg(circ_dist(deg2rad(estimatedAngle), deg2rad(catBound))));
        case 'south'
            boundaryBias = rad2deg(circ_dist(deg2rad(estimatedAngle), deg2rad(catBound)));
            % boundaryBias = abs(rad2deg(circ_dist(deg2rad(estimatedAngle),deg2rad(catBound))));
    end

end
