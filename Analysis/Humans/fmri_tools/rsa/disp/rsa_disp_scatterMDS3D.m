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
