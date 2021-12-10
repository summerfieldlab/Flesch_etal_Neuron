function modelRDMs = mante_gen_modelrdms(params,monitor)
    %
    % generates struct with model RDMs
    %
    % MODELS:
    % gridiness
    % gridiness - rotated
    % factorised
    % factorised - rotated
    %
    % Timo Flesch, 2020

    if ~exist('monitor')
        monitor = 0;
    end
    %% set params and data structures
    
    modelRDMs = struct();
    %% features 
    [b,l] = meshgrid(-2.5:2.5, -2.5:2.5);
    b = b(:);
    l = l(:);
    
    %% generate models
    %% 1. gridiness (parallel planes)        
    xy = [[b,l];[b,l]];     
    xyz = [xy [ones(length(b),1);2.*ones(length(b),1)]];    
    rdm = squareform(pdist(xyz));    
    modelRDMs(1).rdm = rdm;
    modelRDMs(1).name = 'grid';

    %% 2. gridiness (parallel planes)- rotated      
    xy = [[b,l];[b,l]];
    xy(37:end,:) = xy(37:end,:)*[cos(deg2rad(90)),-sin(deg2rad(90));sin(deg2rad(90)),cos(deg2rad(90))]; 
    xyz = [xy [ones(length(b),1);2.*ones(length(b),1)]];    
    rdm = squareform(pdist(xyz));    
    modelRDMs(2).rdm = rdm;
    modelRDMs(2).name = 'grid - rotated';

    %% 3. factorised        
    c1 = [cos(deg2rad(0));sin(deg2rad(0))];
    c2 = [cos(deg2rad(90));sin(deg2rad(90))];
    xyz = [[b,l]*c1*c1',ones(length(b),1);[b,l]*c2*c2',2.*ones(length(b),1)]; 
    
    rdm = squareform(pdist(xyz));    
    modelRDMs(3).rdm = rdm;
    modelRDMs(3).name = 'orthogonal';

    %% 4. factorised - rotated    
    c1 = [cos(deg2rad(0));sin(deg2rad(0))];
    c2 = [cos(deg2rad(90));sin(deg2rad(90))];    
    xy = [[b,l]*c1*c1';[b,l ]*c2*c2'];      
    % xy(1:36,:) = xy(1:36,:)*[cos(deg2rad(90)),-sin(deg2rad(90));sin(deg2rad(90)),cos(deg2rad(90))];     
    xy(37:end,:) = xy(37:end,:)*[cos(deg2rad(90)),-sin(deg2rad(90));sin(deg2rad(90)),cos(deg2rad(90))];     
    xyz = [xy [ones(length(b),1);2.*ones(length(b),1)]];    
    % xyz =xy;
    rdm = squareform(pdist(xyz));    
    modelRDMs(4).rdm = rdm;
    modelRDMs(4).name = 'parallel';

    % 5. only motion
    xy = [b;b];    
    rdm = squareform(pdist(xy));    
    modelRDMs(5).rdm = rdm;
    modelRDMs(5).name = 'only motion';

    % 6. only colour
    xy = [l;l];    
    rdm = squareform(pdist(xy));    
    modelRDMs(6).rdm = rdm;
    modelRDMs(6).name = 'only colour';

    % diagonal model
    c1 = [cos(deg2rad(45));sin(deg2rad(45))];
    c2 = [cos(deg2rad(45));sin(deg2rad(45))];
    xyz = [[b,l]*c1*c1';[b,l]*c2*c2']; 
    
    rdm = squareform(pdist(xyz));    
    modelRDMs(7).rdm = rdm;
    modelRDMs(7).name = 'diagonal';

    % % diagonal model
    % c1 = [cos(deg2rad(45));sin(deg2rad(45))];
    % c2 = [cos(deg2rad(45));sin(deg2rad(45))];
    % xyz = [[b,l]*c1*c1';[b,l]*c2*c2']; 
    
    % rdm = squareform(pdist(xyz));    
    % modelRDMs(8).rdm = rdm;
    % modelRDMs(8).name = 'diagonal2';
   

    if monitor
        %% raw rdms (fig1), mds in 3d (fig2)
        figure(100);set(gcf,'Color','w')
        figure(200);set(gcf,'Color','w')
        for ii = 1:4
            figure(100);
            subplot(2,2,ii)
            aux_showRDM(modelRDMs(ii).rdm);
            title(['\rm' modelRDMs(ii).name]);
            box off;
            axis square;            
            cb = colorbar();
            ylabel(cb,'dissimilarity');
            figure(200);
            subplot(2,2,ii);
            aux_showMDS(modelRDMs(ii).rdm,3);
            title(['\rm' modelRDMs(ii).name]);
            axis square;
        end
        cd(params.dir.figures)
        figure(100);
        set(gcf,'Position',[230,  559, 1112,  732]);
        savefig('modelrdms.fig');
        figure(200);
        set(gcf,'Position',[230,  559, 1112,  732]);
        savefig('modelrdms_mds.fig');

    end
    cd(params.dir.project);
end
