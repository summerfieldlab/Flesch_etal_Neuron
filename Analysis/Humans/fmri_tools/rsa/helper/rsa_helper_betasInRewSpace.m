
function [betasNew,labelsNew] = rsa_helper_betasInRewSpace(betas,labels,rewCode)
  %% [betasNew,labelsNew] = rsa_helper_betasInRewSpace(betas,labels,rewCode)
  %
  % rearranges beta vect such that ordered according to reward, not feature value
  %
  % Timo Flesch, 2019
  % Human Information Processing Lab
  % University of Oxford

  % reference IDs
  [br,lr] = meshgrid(1:5,1:5);
  % new IDs
  switch rewCode
  case 1
    b = br;
    l = lr;
  case 2
    b = fliplr(br);
    l = flipud(lr);
  case 3
    b = br;
    l = flipud(lr);
  case 4
    b = fliplr(br);
    l = lr;
  end
  % vectorise
  b  =  b(:);
  l  =  l(:);
  br = br(:);
  lr = lr(:);

  vectRef = [ones(25,1),lr;ones(25,1).*2,br];
  vectNew = [ones(25,1),l;ones(25,1)*2,b];

  % for each task and reward label, find corresponding betas and append them to matrix
  betasNew  = [];
  labelsNew = [];
  for ctx = 1:2
    for rew = 1:5
      % find betas for current context and reward
      [ii,~] = find(vectNew(vectNew(:,1)==ctx,2)==rew);
      thisBetas = betas(ii,:,:);
      % append to matrix
      betasNew = cat(1,betasNew,thisBetas);
      % append labels
      thisLabels = ['C' num2str(ctx) 'R' num2str(rew)];
      thisLabels = repmat({thisLabels},length(ii),1);
      labelsNew = cat(1,labelsNew,thisLabels);
    end
  end
  labelsNew = repmat(labelsNew,1,size(labels,2));  
end
