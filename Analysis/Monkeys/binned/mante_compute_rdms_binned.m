function results = mante_compute_rdms_binned(params)
    %% MANTE_COMPUTE_RDMS_BINNED(PARAMS)
    %
    % computes rdms separately for early, middle and late window 
    % (217ms each, excluding the 100ms post stimulus interval)
    % 
    %    
    % Timo Flesch, 2020

       
    results = struct();

    for monk = 1:params.analysis.n_monks
        % load data 
        cd([params.dir.data 'monkey' num2str(monk) '/']);
        allunits = dir('*.mat');      
        unitstruct = struct();
        disp('loading data')
        for d = 1:length(allunits)
            load(allunits(d).name);
            unitstruct(d).unit = unit;
        end
        disp('...done');
        
        rdms = NaN(length(params.analysis.t_bins),params.analysis.n_stimdir^2*2,params.analysis.n_stimcol^2*2);
        
        disp('compute rdms for each bin')        
        for (ib = 1:length(params.analysis.t_bins))            
            respmat = helper_compute_respmat(unitstruct,params,params.analysis.t_bins{ib});
            
            % respmat_denoised = helper_denoise(respmat(:,:),12);
            respmat_denoised = respmat(:,:);
            rdms(ib,:,:) =  squareform(pdist(transpose(respmat_denoised),'euclidean'));
            disp(['done with bin ' num2str(ib) '/' num2str(length(params.analysis.t_bins))]);
        end
        results(monk).rdms = rdms;
        results(monk).name = ['monkey ' params.analysis.monknames{monk}];
        results(monk).bins = params.analysis.t_bins;
    end     
    cd(params.dir.results)
    save('monkey_rdms_binned.mat','results');
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