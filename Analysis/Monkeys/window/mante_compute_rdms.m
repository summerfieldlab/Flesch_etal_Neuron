function results = mante_compute_rdms(params,whichneurons)
    %% MANTE_COMPUTE_RDMS()
    %
    % computes rdms for each monkey 

    % adapted from Chris' script 
    % Timo Flesch, 2021
   
    if ~ exist('whichneurons')
        whichneurons = 'all';
    end

    results = struct();

    for monk = 1:params.analysis.n_monks
        cd([params.dir.data 'monkey' num2str(monk) '/']);
        allunits = dir('*.mat');      
        unitstruct = struct();
        disp('loading data')
        for d = 1:length(allunits)
            load(allunits(d).name);
            unitstruct(d).unit = unit;
        end
        disp('...done');        
        if strcmp(whichneurons,'all')
            respmat = helper_compute_respmat(unitstruct,params,params.analysis.t_interval);
            % respmat = helper_denoise(respmat(:,:),12); 
            respmat = respmat(:,:);   
            results(monk).respmat = respmat;
            results(monk).rdm =  squareform(pdist(transpose(respmat)));
            results(monk).name = ['monkey ' params.analysis.monknames{monk}];
            fname = 'monkey_rdms.mat';
        elseif strcmp(whichneurons,'task')
            respmat = helper_compute_respmat(unitstruct,params,params.analysis.t_interval,false,'task');
            % respmat = helper_denoise(respmat(:,:),12);      
            respmat = respmat(:,:);  
            results(monk).respmat = respmat;
            results(monk).rdm =  squareform(pdist(transpose(respmat)));
            results(monk).name = ['monkey ' params.analysis.monknames{monk}];
            fname = 'monkey_rdms_taskselective.mat';
        elseif strcmp(whichneurons,'mixed')
            respmat = helper_compute_respmat(unitstruct,params,params.analysis.t_interval,false,'mixed');
            % respmat = helper_denoise(respmat(:,:),12);   
            respmat = respmat(:,:);     
            results(monk).respmat = respmat;
            results(monk).rdm =  squareform(pdist(transpose(respmat)));
            results(monk).name = ['monkey ' params.analysis.monknames{monk}];
            fname = 'monkey_rdms_mixedselective.mat';
        end


    end 
    % cd(params.dir.results)

    % save(fname,'results');
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