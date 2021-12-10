function aux_showMDS(rdm,ndims)
    %% aux_showMDS(rdm,ndims)
    %
    % projects rdm into ndims-D space 
    % and visualises results as scatter plot
    %
    % Timo Flesch, 2019
    ctxMarkerEdgeCol = [0,11,229; 229,143,0]./255;
    nStims = 72;
    try
        xyz = mdscale(rdm,ndims,'Criterion','strain'); 
    catch
        xyz = mdscale(rdm,ndims,'Criterion','strain','Start','random'); 
        
    end
    disp(size(xyz))
    ctxMarkerCol     = 'w';
    ctxMarkerSize = 20;
    
    scat_branchiness = [5:2:15];
    scat_leafiness = (RDMcolormap_hack(6));
    
    [b,l] = meshgrid(1:6,1:6);
    b = b(:);l = l(:);
    x = xyz(:,1);
    y = xyz(:,2);
    z = xyz(:,3);
    
    % plot grid connecting adjacent points
    aux_disp_grid(xyz(1:nStims/2,:),ctxMarkerEdgeCol(1,:));
    aux_disp_grid(xyz(nStims/2+1:end,:),ctxMarkerEdgeCol(2,:));
    
    % scatterplot (size= dim1, col= dim2)
    [b,l] = meshgrid(1:6,1:6);
    b = [b(:);b(:)];l = [l(:);l(:)];
    for ii = 1:nStims/2
        plot3(x(ii),y(ii),z(ii),'MarkerFaceColor',ctxMarkerCol,'MarkerSize',ctxMarkerSize, ...
        'Marker','square','MarkerEdgeColor',ctxMarkerEdgeCol(1,:),'LineWidth',2);
        hold on;
        plot3(x(ii),y(ii),z(ii),'MarkerFaceColor',scat_leafiness(l(ii),:),'MarkerSize',scat_branchiness(b(ii)), ...
        'Marker','diamond','MarkerEdgeColor','None');
    end
    for ii = nStims/2+1:nStims
        plot3(x(ii),y(ii),z(ii),'MarkerFaceColor',ctxMarkerCol,'MarkerSize',ctxMarkerSize, ...
        'Marker','square','MarkerEdgeColor',ctxMarkerEdgeCol(2,:),'LineWidth',2);
        hold on;
        plot3(x(ii),y(ii),z(ii),'MarkerFaceColor',scat_leafiness(l(ii),:),'MarkerSize',scat_branchiness(b(ii)), ...
        'Marker','diamond','MarkerEdgeColor','None');
    end
    

end