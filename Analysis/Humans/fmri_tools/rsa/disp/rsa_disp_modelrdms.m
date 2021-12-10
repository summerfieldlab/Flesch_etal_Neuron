function rsa_disp_modelrdms()
    %% rsa_disp_modelrdms
    %
    % visualises MDS on modelrdms
    %
    % Timo Flesch, 2020,
    % Human Information Processing Lab,
    % Experimental Psychology Department
    % University of Oxford

    labels = {'01_grid', '02_grid-rotated', '03_orthogonal', '04_parallel'};

    modelRDMs = rsa_corrs_genSLModelRDMs_cval()

    rdms = [];
    for ii = 1:length(labels)
        rdms(:,:,ii) = squeeze(modelRDMs(ii).rdms(1,51:100,1:50));
    end



    for ii = 1:size(rdms,3)
        %TODO
        rdm = squeeze(rdms(:,:,ii));
        figure();set(gcf,'color','w');
        f = gcf;
        [xyz,~] = mdscale(rdm,3,'Criterion','sstress','Options',statset('MaxIter',1000));
        % [xyz] = cmdscale(rdm,2);

        rsa_disp_customGrid3(xyz(1:25,:),[0,11,229]./255,0.5);
        rsa_disp_customGrid3(xyz(26:end,:),[229,143,0]./255,0.5);

        rsa_disp_scatterMDS3D(xyz,f.Number, 'both');
        grid on;
        axis on;
        xticklabels('')
        yticklabels('')
        zticklabels('')
        box off
        set(gcf,'Position',[670   480   621   482]);
        set(gcf,'Color','w')
        set(gca,'LineWidth',1.5)
        % keyboard


        % print(['mds_modelrdms_' labels{ii} ], '-r400','-dpng')
        % close all;
    end




end



function rsa_disp_customGrid3(xyz,edgeCol,edgeWidth)
    %% rsa_disp_customGrid()
    %
    % connects neighbours along the cardinal axes
    %
    % if inputs are provided, the connections depict
    % neighbour-relationships in the undistorted ground-truth space
    %
    % Timo Flesch, 2019
    % University of Oxford
    if ~exist('edgeCol','var')
      edgeCol = [1,1,1].*.8;
    end
  
    if ~exist('edgeWidth','var')
      edgeWidth = 1.5;
    end
  
    [b,l] = meshgrid(1:5,1:5);
    b = b(:);
    l = l(:);
    bl = [b,l];
    if ~exist('xyz','var')
      xyz = [b,l];
    end
  
    for i = 1:4
      for j = 1:4
        p1 = xyz(bl(:,1)==i & bl(:,2)==j,:);
        p2 = xyz(bl(:,1)==i+1 & bl(:,2)==j,:);
        plot3([p1(1),p2(1)],[p1(2),p2(2)],[p1(3),p2(3)],'LineWidth',edgeWidth,'Color',edgeCol);
        hold on;
        p2 = xyz(bl(:,1)==i & bl(:,2)==j+1,:);
        plot3([p1(1),p2(1)],[p1(2),p2(2)],[p1(3),p2(3)],'LineWidth',edgeWidth,'Color',edgeCol);
      end
    end
  
    i = 5;
    for j = 1:4
      p1 = xyz(bl(:,1)==i & bl(:,2)==j,:);
      p2 = xyz(bl(:,1)==i & bl(:,2)==j+1,:);
      plot3([p1(1),p2(1)],[p1(2),p2(2)],[p1(3),p2(3)],'LineWidth',edgeWidth,'Color',edgeCol);
    end
  
    j = 5;
    for i = 1:4
      p1 = xyz(bl(:,1)==i & bl(:,2)==j,:);
      p2 = xyz(bl(:,1)==i+1 & bl(:,2)==j,:);
      plot3([p1(1),p2(1)],[p1(2),p2(2)],[p1(3),p2(3)],'LineWidth',edgeWidth,'Color',edgeCol);
    end
  
    axis off
  end
  



  function rsa_disp_scatterMDS3D(xyz,fIDX,whichTask)
    %% rsa_disp_scatterMDS(xyz,fIDX,whichTask,images)
    %
    % Timo Flesch, 2019
    % Human Information Processing Lab
    % University of Oxford
    
      if ~exist('fIDX','var')
          fIDX = 1000;
      end
      if ~exist('whichTask','var')
        whichTask = 'both';
      end
    
      switch whichTask
      case 'north'
    
        nTrees = 25;
        ctxMarkerEdgeCol = [0,11,229]./255;
    
      case 'south'
        nTrees = 25;
        ctxMarkerEdgeCol = [229,143,0]./255;
    
      case 'both'
        nTrees = 50;
        ctxMarkerEdgeCol = [0,11,229; 229,143,0]./255;
    
      case 'avg'
        nTrees = 25;
        ctxMarkerEdgeCol = 'None';
    
      end
    
      ctxMarkerCol     = 'w';
      ctxMarkerSize = 30;
    
      scat_branchiness = [10:2:18];
      scat_leafiness = [63,39,24; 64,82,21; 65,125,18; 66,168,15; 68,255 10]./255;
    
      [b,l] = meshgrid(1:5,1:5);
      b = b(:);l = l(:);
      x = xyz(:,1);
      y = xyz(:,2);
      z = xyz(:,3);
    
      if strcmp(whichTask,'both')
        [b,l] = meshgrid(1:5,1:5);
        b = [b(:);b(:)];l = [l(:);l(:)];
        for ii = 1:nTrees/2
          plot3(x(ii),y(ii),z(ii),'MarkerFaceColor',ctxMarkerCol,'MarkerSize',ctxMarkerSize, ...
          'Marker','square','MarkerEdgeColor',ctxMarkerEdgeCol(1,:),'LineWidth',2);
          hold on;
          plot3(x(ii),y(ii),z(ii),'MarkerFaceColor',scat_leafiness(l(ii),:),'MarkerSize',scat_branchiness(b(ii)), ...
          'Marker','diamond','MarkerEdgeColor','None');
        end
        for ii = nTrees/2+1:nTrees
          plot3(x(ii),y(ii),z(ii),'MarkerFaceColor',ctxMarkerCol,'MarkerSize',ctxMarkerSize, ...
          'Marker','square','MarkerEdgeColor',ctxMarkerEdgeCol(2,:),'LineWidth',2);
          hold on;
          plot3(x(ii),y(ii),z(ii),'MarkerFaceColor',scat_leafiness(l(ii),:),'MarkerSize',scat_branchiness(b(ii)), ...
          'Marker','diamond','MarkerEdgeColor','None');
        end
    
      else
        [b,l] = meshgrid(1:5,1:5);
        b = b(:); l = l(:);
        for ii = 1:nTrees
          plot3(x(ii),y(ii),z(ii),'MarkerFaceColor',ctxMarkerCol,'MarkerSize',ctxMarkerSize, ...
          'Marker','square','MarkerEdgeColor',ctxMarkerEdgeCol,'LineWidth',2);
          hold on;
          plot3(x(ii),y(ii),z(ii),'MarkerFaceColor',scat_leafiness(l(ii),:),'MarkerSize',scat_branchiness(b(ii)), ...
          'Marker','diamond','MarkerEdgeColor','None');
        end
      end
    
    end
    