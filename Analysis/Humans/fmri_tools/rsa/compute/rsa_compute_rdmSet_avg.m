function rdmSet = rsa_compute_rdmSet_avg(betas, metric)
    %% rsa_compute_rdmSet()
    %
    % computes rdms at single subject level,
    %
    %
    % Timo Flesch, 2019
    % Human Information Processing Lab
    % University of Oxford

    rdmSet = [];

    % compute mahal distances
    switch metric
        case 'crossnobis'
            % disp(' .....  performing cveuclidean distance estimation');
            % do crossvalidated mahal dist for each run (leave one run out cross-validation)
            runIDs = 1:size(betas, 2);

            for runID = runIDs
                bTrain = betas(:, runIDs ~= runID, :);
                bTest = squeeze(betas(:, runID, :));

                for ii = 1:50

                    for jj = 1:50
                        dVTrain = helper_meanCV(squeeze(bTrain(ii, :, :)), squeeze(bTrain(jj, :, :)));
                        dVTest = helper_diffVect(bTest(ii, :), bTest(jj, :));
                        rdm(ii, jj) = dVTrain * dVTest';
                    end

                end

                rdmSet(runID, :, :) = rdm;
            end

        case {'mahalanobis'}
            % disp([' .....  performing ' metric ' distance estimation']);
            % compute rdms with mahal for each run
            for runID = 1:size(betas, 2)
                b = squeeze(betas(:, runID, :));

                for ii = 1:50

                    for jj = 1:50
                        dVTrain = helper_diffVect(b(ii, :), b(jj, :));
                        rdm(ii, jj) = sqrt(dVTrain * dVTrain');
                    end

                end

                rdmSet(runID, :, :) = rdm;
            end

        case {'euclidean', 'correlation', 'cosine'}
            % disp([' .....  performing ' metric ' distance estimation']);
            for runID = 1:size(betas, 2)
                b = squeeze(betas(:, runID, :));
                rdmSet(runID, :, :) = squareform(pdist(b, metric));
            end

    end

    % average across runs
    rdmSet = squeeze(mean(rdmSet, 1));
end

function dv = helper_diffVect(bV1, bV2)
    dv = [bV1 - bV2];
end

function mcv = helper_meanCV(bM1, bM2)
    mcv = helper_diffVect(bM1(1, :), bM2(1, :));

    for ii = 2:size(bM1, 1)
        mcv = mcv + helper_diffVect(bM1(ii, :), bM2(ii, :));
    end

    mcv = mcv ./ size(bM1, 1);
end
