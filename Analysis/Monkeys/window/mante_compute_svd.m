function singvals = mante_compute_svd(params,results)
    %% mante_compute_svd(params)
    %
    % computes singular varlues of response matrix
    % as well as % explained variance
    %
    % Timo Flesch, 2021
    singvals = struct();
    for monk = 1:2
    [u,s,v] = svd(results(monk).respmat);
    singvals(monk).sv = diag(s)
    singvals(monk).var = singvals(monk).sv.^2 ./sum(singvals(monk).sv.^2);
    singvals(monk).cumvar = cumsum(singvals(monk).var)

end