function rdmSet = rsa_compute_rdmSet_cval(betas, metric)
    %% rsa_compute_rdmSet()
    %
    % computes rdms at single subject level,
    % ignores within run dissimilarities (NaN)
    % warning: RDMs can be very large (6 runs, 50conditions per run: 300x300 entries)
    %
    % Timo Flesch, 2019
    % Human Information Processing Lab
    % University of Oxford

    rdmSet = [];
    % concatenate runs:
    b = squeeze(betas(:, 1, :));

    for runID = 2:size(betas, 2)
        b = cat(1, b, squeeze(betas(:, runID, :)));
    end

    % compute mahal distances
    switch metric
        case {'mahalanobis'}

            for ii = 1:size(b, 1)

                for jj = 1:size(b, 1)
                    dVTrain = helper_diffVect(b(ii, :), b(jj, :));
                    rdm(ii, jj) = sqrt(dVTrain * dVTrain');
                end

            end

            rdmSet = rdm;
        case {'euclidean', 'correlation', 'cosine'}
            rdmSet = squareform(pdist(b, metric));
    end

    % nan all within run comparisons
    nConds = size(betas, 1);
    nRuns = size(betas, 2);

    for iiRun = 1:nConds:(nConds * nRuns)
        rdmSet(iiRun:iiRun + nConds - 1, iiRun:iiRun + nConds - 1) = NaN;
    end

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
