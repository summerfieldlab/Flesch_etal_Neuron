function ort_rdmSet = orthogonalize_modelRDMs(rdmSet)
	%% ORTRDMSET = ORTHOGONALIZE_MODELRDMS(RDMSET)
	%
	% applies recursive gram schmidt orthogonalisation on RDMs
	% c.f. Spitzer et al, 2017 
	%
	% Timo Flesch, 2017
	
    rdmVects = [];
    % iterate through rdms
    for modID=1:size(rdmSet,1)
    	% vectorize RDM
        rdmVects(:,modID)=vectorizeRDM(squeeze(rdmSet(modID,:,:)));
        % mean-center RDM
        rdmVects(:,modID)=rdmVects(:,modID)-mean(rdmVects(:,modID)); 
    end

    % apply recursive orthogonalisation
    for modID=1:size(rdmVects,2)
    	% define orthogonalisation order (current last)
        ortho_order = [find((1:size(rdmVects,2))~=modID) modID];
        % orthogonalize
        ort_rdmVects = spm_orth(rdmVects(:,ortho_order));
        % save the last entry (corresponding to the current candidate model)
        ort_rdmSet(modID,:,:) = squareform(ort_rdmVects(:,end));
    end