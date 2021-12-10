function rsa_disp_customGrid(xy,edgeCol,edgeWidth)
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
  if ~exist('xy','var')
    xy = [b,l];
  end

  for i = 1:4
    for j = 1:4
      p1 = xy(bl(:,1)==i & bl(:,2)==j,:);
      p2 = xy(bl(:,1)==i+1 & bl(:,2)==j,:);
      plot([p1(1),p2(1)],[p1(2),p2(2)],'LineWidth',edgeWidth,'Color',edgeCol);
      hold on;
      p2 = xy(bl(:,1)==i & bl(:,2)==j+1,:);
      plot([p1(1),p2(1)],[p1(2),p2(2)],'LineWidth',edgeWidth,'Color',edgeCol);
    end
  end

  i = 5;
  for j = 1:4
    p1 = xy(bl(:,1)==i & bl(:,2)==j,:);
    p2 = xy(bl(:,1)==i & bl(:,2)==j+1,:);
    plot([p1(1),p2(1)],[p1(2),p2(2)],'LineWidth',edgeWidth,'Color',edgeCol);
  end

  j = 5;
  for i = 1:4
    p1 = xy(bl(:,1)==i & bl(:,2)==j,:);
    p2 = xy(bl(:,1)==i+1 & bl(:,2)==j,:);
    plot([p1(1),p2(1)],[p1(2),p2(2)],'LineWidth',edgeWidth,'Color',edgeCol);
  end

  axis off
end
