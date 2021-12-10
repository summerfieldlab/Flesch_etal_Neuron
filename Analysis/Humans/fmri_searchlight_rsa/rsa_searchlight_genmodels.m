function modelRDMs = rsa_searchlight_genmodels()
    %
    % sets up struct with model RDMs for
    % final searchlight RSA
    %
    % MODELS:
    % gridiness
    % gridiness - rotated
    % factorised
    % factorised - rotated
    % only branchiness
    % only leafiness
    % diagonal
    %
    %
    % Timo Flesch, 2019,
    % Human Information Processing Lab,
    % Experimental Psychology Department
    % University of Oxford

    %% set params and data structures
    params = rsa_searchlight_params();
    modelRDMs = struct();

    %% import data
    % load behavioural data
    rsData = load([params.dir.behavDir 'rsData_scan.mat']);
    fns = fieldnames(rsData);
    rsData = rsData.(fns{1});

    %% generate models

    %% 1. gridiness (parallel planes )
    [b, l] = meshgrid(-2:2, -2:2);
    b = b(:);
    l = l(:);
    % set up grid
    respVect = [[l, b]; [l, b]]; %l=north,b=south
    % add offset
    respVect = [respVect [ones(length(b), 1); 2 .* ones(length(b), 1)]];
    % generate rdm
    rdm = squareform(pdist(respVect));
    % no diff between subs, thus just replicate n times
    for (ii = 1:length(params.num.goodSubjects))
        subID = params.num.goodSubjects(ii);
        %expand RDM
        rdm2 = helper_expandRDM(rdm, params.num.runs);
        rdms(ii, :, :) = rdm2;
    end

    modelRDMs(1).rdms = rdms;
    modelRDMs(1).name = 'gridiness';

    %% 2. gridiness (parallel planes)- rotated TODO: rotate one of the ctx vectors by 90°
    [b, l] = meshgrid(-2:2, -2:2);
    b = b(:);
    l = l(:);

    for (ii = 1:length(params.num.goodSubjects))
        subID = params.num.goodSubjects(ii);
        boundID = rsData(subID).code(end);
        % create grid
        xy = [[l, b]; [l, b]]; %l=north,b=south
        % rotate appropriately
        if boundID == 3 || boundID == 4
            xy(26:end, :) = xy(26:end, :) * [cos(deg2rad(270)), -sin(deg2rad(270)); sin(deg2rad(270)), cos(deg2rad(270))]; % rotate vector
        else
            xy(26:end, :) = xy(26:end, :) * [cos(deg2rad(90)), -sin(deg2rad(90)); sin(deg2rad(90)), cos(deg2rad(90))]; % rotate vector
        end

        % add offset
        xy = [xy [ones(length(b), 1); 2 .* ones(length(b), 1)]];
        % compute rdm
        rdm = squareform(pdist(xy));
        %expand RDM
        rdm2 = helper_expandRDM(rdm, params.num.runs);
        % store results
        rdms(ii, :, :) = rdm2;
    end

    modelRDMs(2).rdms = rdms;
    modelRDMs(2).name = 'gridiness - rotated';

    %% 3. factorised
    [b, l] = meshgrid([-2:2], [-2:2]);
    b = b(:);
    l = l(:);
    % respVect = [b,zeros(length(b),1);zeros(length(l),1),l];
    c1 = [cos(deg2rad(0)); sin(deg2rad(0))];
    c2 = [cos(deg2rad(90)); sin(deg2rad(90))];
    respVect = [[l, b] * c1 * c1', ones(length(b), 1); [l, b] * c2 * c2', 2 .* ones(length(b), 1)]; %l=north,b=south
    rdm = squareform(pdist(respVect));

    for (ii = 1:length(params.num.goodSubjects))
        subID = params.num.goodSubjects(ii);
        %expand RDM
        rdm2 = helper_expandRDM(rdm, params.num.runs);
        rdms(ii, :, :) = rdm2;
    end

    modelRDMs(3).rdms = rdms;
    modelRDMs(3).name = 'factorised';

    %% 4. factorised - rotated
    [b, l] = meshgrid([-2:2], [-2:2]);
    b = b(:);
    l = l(:);
    c1 = [cos(deg2rad(0)); sin(deg2rad(0))];
    c2 = [cos(deg2rad(90)); sin(deg2rad(90))];

    for (ii = 1:length(params.num.goodSubjects))
        subID = params.num.goodSubjects(ii);
        boundID = rsData(subID).code(end);
        xy = [[l, b] * c1 * c1'; [l, b] * c2 * c2'];

        if boundID == 3 || boundID == 4
            xy(26:end, :) = xy(26:end, :) * [cos(deg2rad(270)), -sin(deg2rad(270)); sin(deg2rad(270)), cos(deg2rad(270))]; % rotate vector
        else
            xy(26:end, :) = xy(26:end, :) * [cos(deg2rad(90)), -sin(deg2rad(90)); sin(deg2rad(90)), cos(deg2rad(90))]; % rotate vector
        end

        % add offset
        xy = [xy [ones(length(b), 1); 2 .* ones(length(b), 1)]];
        % compute rdm
        rdm = squareform(pdist(xy));
        %expand RDM
        rdm2 = helper_expandRDM(rdm, params.num.runs);
        rdms(ii, :, :) = rdm2;
    end

    modelRDMs(4).rdms = rdms;
    modelRDMs(4).name = 'factorised -rotated';

    %% 5. only branchiness
    xy = [b; b];
    rdm = squareform(pdist(xy));

    for (ii = 1:length(params.num.goodSubjects))
        subID = params.num.goodSubjects(ii);
        %expand RDM
        rdm2 = helper_expandRDM(rdm, params.num.runs);
        rdms(ii, :, :) = rdm2;
    end

    modelRDMs(5).rdms = rdms;
    modelRDMs(5).name = 'only branchiness';

    %% 6. only leafiness
    xy = [l; l];
    rdm = squareform(pdist(xy));

    for (ii = 1:length(params.num.goodSubjects))
        subID = params.num.goodSubjects(ii);
        %expand RDM
        rdm2 = helper_expandRDM(rdm, params.num.runs);
        rdms(ii, :, :) = rdm2;
    end

    modelRDMs(6).rdms = rdms;
    modelRDMs(6).name = 'only leafiness';

    %% diagonal model

    for (ii = 1:length(params.num.goodSubjects))
        subID = params.num.goodSubjects(ii);
        boundID = rsData(subID).code(end);
        % xy = [[l, b] * c1 * c1'; [l, b] * c2 * c2'];

        if boundID == 1 || boundID == 2
            c1 = [cos(deg2rad(135)); sin(deg2rad(135))];
            c2 = [cos(deg2rad(135)); sin(deg2rad(135))];
            xy = [[l, b] * c1 * c1'; [l, b] * c2 * c2'];
        else
            c1 = [cos(deg2rad(45)); sin(deg2rad(45))];
            c2 = [cos(deg2rad(45)); sin(deg2rad(45))];
            xy = [[l, b] * c1 * c1'; [l, b] * c2 * c2'];
        end

        % add offset
        xy = [xy [ones(length(b), 1); 2 .* ones(length(b), 1)]];
        % compute rdm
        rdm = squareform(pdist(xy));
        %expand RDM
        rdm2 = helper_expandRDM(rdm, params.num.runs);
        rdms(ii, :, :) = rdm2;
    end

    modelRDMs(7).rdms = rdms;
    modelRDMs(7).name = 'diagonal';

end

function rdm = helper_expandRDM(rdm, nRuns)

    nConds = size(rdm, 2);
    rdm = repmat(rdm, [nRuns, nRuns]);

    for iiRun = 1:nConds:(nConds * nRuns)
        rdm(iiRun:iiRun + nConds - 1, iiRun:iiRun + nConds - 1) = NaN;
    end

end
