function pipeline_analyse_and_plot_arena(repoDir)
    %% PIPELINE_ANALYSE_ARENA(REPODIR)
    %
    % wrapper function to perform analysis of arena task data
    %
    % INPUT:
    % repoDir: path to this repository
    %
    % Timo Flesch, 2018
    % Human Information Processing Lab, Experimental Psychology Department
    % University of Oxford

    % load and import raw data
    dissimData = dissim_getAllData([repoDir 'Data/Humans/part1_arena/']);

    %% RSA
    % compute RDMs
    subjectRDMs = dissim_computeRDMs(dissimData);

    % compute model rdms (branchiness,leafiness,gridiness,pixel dissimilarity):
    pixelRDMs = dissim_computePixelRDMs(dissimData, [repoDir 'Experiments/Humans/part1_arena/stims/']);
    modelRDMs = dissim_computeModelRDMs();
    % compute gridiness scores:
    tauas = dissim_computeRDMCorrelations_features(modelRDMs, pixelRDMs, subjectRDMs);

    % visualise results:
    % .. model correlations:
    dissim_dispCorrelationResults(tauas, 1);
    % .. distribution of gridiness scores wrt median of pnas study participants (0.1843, n=241)
    dissim_disp_prior_histogram(tauas);
    % .. mds on low, med and high gridiness score participants
    dissim_disp_LowMedHiGP(dissimData, subjectRDMs, tauas, [repoDir 'Experiments/Humans/part1_arena/stims/']);

end

function dissimData = dissim_getAllData(dataDir)
    % imports and transforms single subject data
    % returns data struct
    if (~exist('dataDir', 'var'))
        disp('please set file directory!');
        return;
    end

    dissimData = struct();
    subjects = [1:32];

    % load and add data
    for ii = 1:length(subjects)
        subj = subjects(ii);
        % tmp   = load([dataDir '/s' num2str(subj)]);
        fName = mk_fileName(subj);
        tmp = load([dataDir fName]);
        names = fieldnames(tmp);
        data = tmp.(names{1});

        dissimData(ii).params = data.dissimParams;
        dissimData(ii).stats = data.dissimStats;

        % DISSIM - ALL
        dissimData(ii).data = [];
        dissimData(ii).structure = 'trial | leaf | branch | x_orig | y_orig | x_final | y_final';
        dissimData(ii).data(:, 1) = data.dissimData.pre_trialID;
        dissimData(ii).data(:, 2) = data.dissimData.pre_stimLeafLevel;
        dissimData(ii).data(:, 3) = data.dissimData.pre_stimBranchLevel;
        dissimData(ii).data(:, 4) = data.dissimData.pre_stimCoords_Orig(:, 1);
        dissimData(ii).data(:, 5) = data.dissimData.pre_stimCoords_Orig(:, 2);
        dissimData(ii).data(:, 6) = data.dissimData.pre_stimCoords_Final(:, 1);
        dissimData(ii).data(:, 7) = data.dissimData.pre_stimCoords_Final(:, 2);

        % EXEMPLARS - ALL
        dissimData(ii).stimuli = data.dissimData.pre_stimNames;
    end

end

function rdmSet = dissim_computeRDMs(data)
    % iterates over subjects and computes trial-wise rdms
    %% MAIN
    rdmSet = [];
    % loop through subjects and trials
    for ii = 1:length(data)

        for jj = 1:max(data(1).data(:, 1))
            coords = data(ii).data(data(ii).data(:, 1) == jj, 6:7);
            dists = pdist(coords);
            dists = dists ./ max(dists);
            rdm = squareform(dists);
            rdmSet(ii, jj, :, :) = rdm;
        end

    end

end

function rdmSet = dissim_computePixelRDMs(data, imagePath)
    % computes pixel-based RDMS
    % uses the exemplars that were shown to the participants
    % returns subject-x-trial-x-dim1-x-dim2 set of rdms

    numTrials = max(data(1).data(:, 1));
    rdmSet = [];

    for ii = 1:length(data)
        idces = 1:25;

        for jj = 1:numTrials
            % load images
            stimMat = helper_loadImages(imagePath, data(ii).stimuli(idces), 0);
            % vectorize the images
            stimMat = double(stimMat(1:25, :));
            % compute rdms:
            rdmSet(ii, jj, :, :) = squareform(pdist(stimMat));
            idces = idces + 25;
            stimMat = [];
        end

    end

end

function rdmSet = dissim_computeModelRDMs()
    % computes model rdms based on
    % - first feature dimension
    % - second feature dimension
    % - 2d grid
    % assuming linear spacing of categorical arrangement

    rdmSet = [];

    [branch, leaf] = meshgrid(1:5, 1:5);
    branch = branch(:);
    leaf = leaf(:);
    % branch model:
    rdmSet(1, :, :) = squareform(pdist(branch));

    % leaf model:
    rdmSet(2, :, :) = squareform(pdist(leaf));

    % grid model:
    rdmSet(3, :, :) = squareform(pdist([leaf, branch]));
end

function tauas = dissim_computeRDMCorrelations_features(modelRDMs, pixelRDMs, subjectRDMs)
    % computes correlation between feature based model rdms and subject rdms
    % for arena task

    %orthogonalise models
    for subID = 1:size(pixelRDMs, 1)
        rdms = [];

        for tID = 1:size(pixelRDMs, 2)

            for mID = 1:size(modelRDMs, 1)
                rdms(mID, tID, :, :) = squeeze(modelRDMs(mID, :, :));
            end

            rdms(size(modelRDMs, 1) + 1, tID, :, :) = squeeze(pixelRDMs(subID, tID, :, :));
            rdms_orth(:, tID, :, :) = orthogonalize_modelRDMs(squeeze(rdms(:, tID, :, :)));

            pixelRDMs(subID, tID, :, :) = squeeze(rdms_orth(end, tID, :, :));
        end

    end

    for mID = 1:size(modelRDMs, 1) - 1
        modelRDMs(mID, :, :) = squeeze(mean(rdms_orth(mID, :, :, :), 2));
    end

    tauas = [];

    % average rdms across trials:
    subjectRDMs = squeeze(mean(subjectRDMs, 2));
    pixelRDMs = squeeze(mean(pixelRDMs, 2));

    % iterate through subs:
    for ii = 1:size(subjectRDMs, 1) % for all subjects

        for jj = 1:size(modelRDMs, 1)
            tauas(ii, jj) = rankCorr_Kendall_taua(vectorizeRDM(squeeze(modelRDMs(jj, :, :))), vectorizeRDM(squeeze(subjectRDMs(ii, :, :))));
        end

        % stimuli (pixel rdms)
        tauas(ii, size(modelRDMs, 1) + 1, 1) = rankCorr_Kendall_taua(vectorizeRDM(squeeze(pixelRDMs(ii, :, :))), vectorizeRDM(squeeze(subjectRDMs(ii, :, :))));
    end

end

function dissim_dispCorrelationResults(tauas_features, doStats)
    %% DISSIM_DISPCORRELATIONRESULTS(TAUAS_FEATURES, DOSTATS)
    %
    % displays results of correlation analysis
    %
    % Timo Flesch, 2018
    % Human Information Processing Lab,
    % Experimental Psychology Department
    % University of Oxford

    if ~(exist('doStats', 'var'))
        doStats = 1;
    end

    modelNames = {'branch', 'leaf', 'grid', 'pixel'};

    sem = @(X, dim) std(X, 0, dim) ./ sqrt(size(X, dim));
    colvals = linspace(0.2, .8, 4);

    t_mu = squeeze(mean(tauas_features, 1));
    t_er = squeeze(sem(tauas_features, 1));

    figure(); set(gcf, 'Color', 'w');

    b1 = bar(1, t_mu(1), 'LineWidth', 1.5, 'EdgeColor', 'none');
    b1.FaceColor = [1, 1, 1] .* colvals(1);
    hold on;
    sc = scatter(ones(size(tauas_features, 1), 1), tauas_features(:, 1));
    sc.MarkerEdgeColor = [1, 1, 1] .* 0.4;
    sc.MarkerFaceColor = [1, 1, 1] .* 0.9;
    b2 = bar(2, t_mu(2), 'LineWidth', 1.5, 'EdgeColor', 'none');
    b2.FaceColor = [1, 1, 1] .* colvals(2);
    sc = scatter(ones(size(tauas_features, 1), 1) .* 2, tauas_features(:, 2));
    sc.MarkerEdgeColor = [1, 1, 1] .* 0.4;
    sc.MarkerFaceColor = [1, 1, 1] .* 0.9;
    b3 = bar(3, t_mu(3), 'LineWidth', 1.5, 'EdgeColor', 'none');
    b3.FaceColor = [1, 1, 1] .* colvals(3);
    sc = scatter(ones(size(tauas_features, 1), 1) .* 3, tauas_features(:, 3));
    sc.MarkerEdgeColor = [1, 1, 1] .* 0.4;
    sc.MarkerFaceColor = [1, 1, 1] .* 0.9;
    b4 = bar(4, t_mu(4), 'LineWidth', 1.5, 'EdgeColor', 'none');
    b4.FaceColor = [1, 1, 1] .* colvals(4);
    sc = scatter(ones(size(tauas_features, 1), 1) .* 4, tauas_features(:, 4));
    sc.MarkerEdgeColor = [1, 1, 1] .* 0.4;
    sc.MarkerFaceColor = [1, 1, 1] .* 0.9;

    hold on;

    errorbar(t_mu, t_er, 'LineStyle', 'none', 'Color', 'k', 'MarkerSize', 5, 'LineWidth', 2.5);

    xlabel('\bf Model-RDM', 'FontName', 'Arial', 'FontSize', 10);
    ylabel({'\bf Correlation (\tau_{a})'}, 'FontName', 'Arial', 'FontSize', 10);

    if doStats

        for ii = 1:3

            for jj = (ii + 1):4
                [pval, ~, stats] = ranksum(tauas_features(:, ii), tauas_features(:, jj), 'method', 'approximate');
                d = compute_cohensD('t2', mean(tauas_features(:, ii)), std(tauas_features(:, ii), 0, 1), mean(tauas_features(:, jj)), std(tauas_features(:, jj), 0, 1));
                fprintf([modelNames{ii} ' != ' modelNames{jj} ':\t p= ' num2str(round(pval * 1000) / 1000) ',\t zval ' num2str(stats.zval) '\t d=' num2str(d) '\n']);
                sigstar([ii, jj], pval);
            end

        end

    end

    box off;
    set(gca, 'XTick', 1:4);
    set(gca, 'XTickLabel', modelNames);

end

function dissim_disp_prior_histogram(tauas_features)
    %% DISSIM_DISP_PRIOR_HISTOGRAM
    %
    % plots histograms of gridiness priors
    % wrt median of data reported in flesch et al, 2018 (tau_grid = 0.1843, n=241)

    %% ... + all participants
    figure(); set(gcf, 'Color', 'w');

    taus_all = tauas_features(:, 3);

    cf2 = histogram(taus_all, 'FaceColor', [1 0 0], 'EdgeColor', [0 0 0], 'NumBins', 5);
    hold on
    m1 = plot([0.1843, 0.1843], get(gca, 'YLim'), 'b-', 'LineWidth', 3);
    m2 = plot([median(taus_all), median(taus_all)], get(gca, 'YLim'), 'r-', 'LineWidth', 3);
    legend([m1, cf2, m2], {['Median PNAS: ' num2str(0.18)], ['Hist this study (n=' num2str(length(taus_all)) ')'], ['Median this study: ' num2str(round(median(taus_all), 2))]}, 'Location', 'NorthEastOutside');
    legend boxoff;
    box off;
    xlabel('Grid Prior (Kendall''s \tau_{a})');
    ylabel('count');

end

function dissim_disp_LowMedHiGP(data, rdmSet, tauas_features, imagePath)
    %% DISSIM_DISP_LOWMEDHIGP(DATA,RDMSET,TAUAS_FEATURES)
    %
    % displays mds of ratings, separately
    % for subjects with low, medium and high priors

    [taus, idces] = sort(tauas_features(:, 3));
    binSize = floor(length(taus) / 3);
    taus_lo = taus(1:binSize);
    rdm_lo = rdmSet(idces(1:binSize), :, :, :);
    taus_med = taus(binSize + 1:2 * binSize);
    rdm_med = rdmSet(idces(binSize + 1:2 * binSize), :, :, :);
    taus_hi = taus(2 * binSize + 1:end);
    rdm_hi = rdmSet(idces(2 * binSize + 1:end), :, :, :);

    fileNames = data(2).stimuli;
    idces = 1:25;

    figure(); set(gcf, 'Color', 'w');
    subplot(1, 3, 1);
    mdsSet = dissim_performMDS(rdm_lo);
    tmp = dissim_generateArrangementImg(fileNames, mdsSet, idces, imagePath);
    imshow(tmp);
    axis square;
    xlabel([]);
    ylabel([]);
    title({'Low Prior'; [num2str(round(min(taus_lo), 2)) '-' num2str(round(max(taus_lo), 2))]})

    subplot(1, 3, 2);
    mdsSet = dissim_performMDS(rdm_med);
    tmp = dissim_generateArrangementImg(fileNames, mdsSet, idces, imagePath);
    imshow(tmp);
    axis square;
    xlabel([]);
    ylabel([]);
    title({'Medium Prior'; [num2str(round(min(taus_med), 2)) '-' num2str(round(max(taus_med), 2))]})

    subplot(1, 3, 3);
    mdsSet = dissim_performMDS(rdm_hi);
    tmp = dissim_generateArrangementImg(fileNames, mdsSet, idces, imagePath);
    imshow(tmp);
    axis square;
    xlabel([]);
    ylabel([]);
    title({'High Prior'; [num2str(round(min(taus_hi), 2)) '-' num2str(round(max(taus_hi), 2))]})

end

function [mdsSet] = dissim_performMDS(rdmSet)
    % performs mds on group-level rdms

    %% main
    mdsSet = [];
    rdmSet = squeeze(mean(mean(rdmSet, 2), 1));
    mdsSet = mdscale(rdmSet, 2, 'Criterion', 'metricstress');

end

function G = dissim_generateArrangementImg(stimuli, data, trialID, imagePath)
    %% DISSIM_GENERATEARRANGEMENTIMG(STIMULI,DATA,TRIALID)
    %
    % generates scatter image with stimulus thumbnails
    % embedding algorithm from a tsne visualisation script by Andre Karpathy
    %
    % NOTE: change comment on four lines below depending on wether you'd like
    % to visualize actual ratings or MDS plots
    %
    % Input:
    % stimuli: cell with file names
    % data:    matrix with coordinates etc
    % trialID: trialid to index correct files and mat rows
    %
    % Output:
    % scatterimage: rgb image of final arrangement

    % for rdm mds visualisation:
    idces = trialID;
    y = data;

    images = helper_loadImages(imagePath, stimuli(idces));

    % center the coordinates
    y = bsxfun(@minus, y, min(y));
    y = bsxfun(@rdivide, y, max(y));

    %% MAIN
    S = 1200; % size of full embedding image
    G = ones(S, S, 3, 'uint8') .* 150;
    s = 80; % size of every single image

    Ntake = size(y, 1);

    for i = 1:Ntake

        if mod(i, 100) == 0
            fprintf('%d/%d...\n', i, Ntake);
        end

        % location
        a = ceil(y(i, 1) * (S - s) + 1);
        b = ceil(y(i, 2) * (S - s) + 1);
        a = a - mod(a - 1, s) + 1;
        b = b - mod(b - 1, s) + 1;

        if G(a, b, 1) ~= 150
            continue % spot already filled
        end

        I = squeeze(images(i, :, :, :));
        if size(I, 3) == 1, I = cat(3, I, I, I); end
        I = imresize(I, [s, s]);

        G(a:a + s - 1, b:b + s - 1, :) = I;

    end

end
