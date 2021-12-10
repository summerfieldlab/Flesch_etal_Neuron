function b = aux_regress_rdm(modelRDMs,yRDM)
    %% aux_regress_rdm(modelRDMs,yRDM)
    %
    % regresses (brain) RDM against several model RDMs 
    % (multiple linear regression)
    %
    % Timo Flesch
    
    dmat = helper_construct_designMatrix(modelRDMs);    
    y = transpose(nanzscore(vectorizeRDM(yRDM)));    
    
    b = regress(y,dmat);
    
end 


function dmat = helper_construct_designMatrix(modelRDMs)
    dmat = [];
    for modID = 1:length(modelRDMs)
        dmat(:,modID) = nanzscore(vectorizeRDM(squeeze(modelRDMs(modID).rdm)));
    end
end
