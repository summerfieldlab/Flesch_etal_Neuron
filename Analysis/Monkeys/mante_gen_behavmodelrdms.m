function myModel = mante_gen_behavmodelrdms(modIDX)
    %% MYMODEL = mante_gen_behavmodelrdms(modIDX)
    %
    % computes model rdm for branch-leaf cl experiment
    % with one relevant and one irrelevant dimension
    %
    % INPUT:
    %	- modIDX: model index (1 to 4)
    %
    % OUTPUT:
    %	- the desired model rdm + choice mat
    %
    % (c) Timo Flesch, 2017
    % Summerfield Lab, Experimental Psychology Department, University of Oxford
    myModel = struct();
    
    %% main
    % 1. compute choice matrix:
    switch modIDX
    case 1
        choiceMat = repmat([0;0;0;1;1;1],1,6); % cardinal north
    case 2
        choiceMat = repmat([0,0,0,1,1,1],6,1); % cardinal south (ortho diagonal)
    case 3
        choiceMat = tril(ones(6,6));           % diagonal north (ortho cardinal)
        choiceMat(1:length(choiceMat)+1:end) = 0.5;
        choiceMat = fliplr(choiceMat);
    case 4
        choiceMat = tril(ones(6,6));           % diagonal south
        choiceMat(1:length(choiceMat)+1:end) = 0.5;
        choiceMat = rot90(choiceMat,2);
    
    
    end
    
    
    % 2. compute rdm:
    myModel.choiceMat = choiceMat;
    myModel.choiceRDM = squareform(pdist(choiceMat(:)));
    end
    