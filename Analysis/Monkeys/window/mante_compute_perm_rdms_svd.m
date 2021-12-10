function results = mante_compute_perm_rdms_svd(params,whichneurons)
    %% mante_compute_perm_rdms()
    % 
    % computes rdms from randomly permuted data 
    % to create a null distribution of rdms 
    %
    % labels are permuted at single neuron level 
    %
    % Timo Flesch, 2020

   
    
    results = struct();

    if ~ exist('whichneurons')
        whichneurons = 'all';
    end

    for monk = 1:params.analysis.n_monks
        % load data, (do this once, ow n_perms * n_units read ops.......)
        cd([params.dir.data 'monkey' num2str(monk) '/']);
        allunits = dir('*.mat');
        unitstruct = struct();
        disp('loading data')
        for d = 1:length(allunits)
            load(allunits(d).name);
            unitstruct(d).unit = unit;
        end
        % init rdm set 
        results(monk).rdms = nan(params.stats.n_perms,params.analysis.n_stimdir^2*2,params.analysis.n_stimdir^2*2,params.analysis.n_stimdir^2*2);
        % rAndOmiZeeee
        disp('starting permutation test')
        tic
        for perm = 1:params.stats.n_perms;
            if strcmp(whichneurons,'all')
                respmat = helper_compute_respmat(unitstruct,params,params.analysis.t_interval,true);
                fname ='monkey_perm_rdms_svd.mat';
            elseif strcmp(whichneurons,'task')
                respmat = helper_compute_respmat(unitstruct,params,params.analysis.t_interval,true,'task');
                fname = 'monkey_perm_rdms_taskselective_svd.mat';
            elseif strcmp(whichneurons,'mixed')
                respmat = helper_compute_respmat(unitstruct,params,params.analysis.t_interval,true,'mixed');
                fname = 'monkey_perm_rdms_mixedselective_svd.mat';
            end           

            for sv = 1:72
                respmat_svd = helper_denoise(respmat(:,:),sv);
                results(monk).rdms(perm,sv,:,:) =  squareform(pdist(transpose(respmat_svd)));
            end
            if mod(perm,10)==0
                toc
                disp(['finished ' num2str(perm) '/' num2str(params.stats.n_perms) ' permutations']);
                tic
            end
        end
        results(monk).name = ['monkey ' params.analysis.monknames{monk}];
    end
    cd(params.dir.results)
    save(fname,'results','-v7.3');
    cd(params.dir.project);
end

function xData_reduced = helper_denoise(xData, nDims)
    xData = xData-mean(xData,1);
    [U,S,V] = svd(xData);
    S_reduced = 0*S;
    for ii = 1:nDims
        S_reduced(ii,ii) = S(ii,ii);
    end

    xData_reduced = U*S_reduced*V';
end