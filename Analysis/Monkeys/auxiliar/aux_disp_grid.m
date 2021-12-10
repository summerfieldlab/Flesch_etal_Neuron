function aux_disp_grid(xyz,edgeCol,edgeWidth)
  %% rsa_disp_customGrid(xyz,edgeCol,edgeWidth)
  %
  % draws lines between adjacent points on a plane
  %
  % Timo Flesch, 2019
  % University of Oxford

  if ~exist('edgeCol','var')
    edgeCol = [1,1,1].*.8;
  end

  if ~exist('edgeWidth','var')
    edgeWidth = 1.5;
  end

  [l,b] = meshgrid(1:6,1:6);
  b = b(:);
  l = l(:);
  bl = [b,l];
  if ~exist('xyz','var')
    xyz = [b,l];
  end

  for i = 1:5
    for j = 1:5
      p1 = xyz(bl(:,1)==i & bl(:,2)==j,:);
      p2 = xyz(bl(:,1)==i+1 & bl(:,2)==j,:);
      plot3([p1(1),p2(1)],[p1(2),p2(2)],[p1(3),p2(3)],'LineWidth',edgeWidth,'Color',edgeCol);
      hold on;
      p2 = xyz(bl(:,1)==i & bl(:,2)==j+1,:);
      plot3([p1(1),p2(1)],[p1(2),p2(2)],[p1(3),p2(3)],'LineWidth',edgeWidth,'Color',edgeCol);
    end
  end

  i = 6;
  for j = 1:5
    p1 = xyz(bl(:,1)==i & bl(:,2)==j,:);
    p2 = xyz(bl(:,1)==i & bl(:,2)==j+1,:);
    plot3([p1(1),p2(1)],[p1(2),p2(2)],[p1(3),p2(3)],'LineWidth',edgeWidth,'Color',edgeCol);
  end

  j = 6;
  for i = 1:5
    p1 = xyz(bl(:,1)==i & bl(:,2)==j,:);
    p2 = xyz(bl(:,1)==i+1 & bl(:,2)==j,:);
    plot3([p1(1),p2(1)],[p1(2),p2(2)],[p1(3),p2(3)],'LineWidth',edgeWidth,'Color',edgeCol);
  end

  % axis off
end
