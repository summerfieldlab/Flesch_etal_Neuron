function modelRDMs = rsa_corrs_genModelRDMs_cval()
  %%  rsa_corrs_genModelRDMs_cval
  % generates model RDMs for all subjects
  % , arranged for crossvalidation between runs
  %
  % models:
  % - pixel space (using the images seen by subjects)
  % - dissimilarity ratings (as provided by subjects)
  % - contexts x dimension 1D (e.g. choices: branchy5 in ctxA == leafy5 in ctxB)
  % - contexts x dimension 2D (e.g. orthogonal vectors in 2d Euclidean space)
  % - categorical contexts x leafiness x branchiness 1D
  % - categorical contexts x leafiness x branchiness 2D
  % - categorical linear model (diagonal boundary) 1D
  % - categorical linear model (diagonal boundary) 2D
  % - context (north vs south): main effect of context
  %
  % returns a subject x model matrix as well as a cell struct with model labels
  % Timo Flesch, 2019

  % notes:
  % - ctx1: north, ctx2: south.
  % leafiness: ctx1
  % branchinessL ctx2

  %% set params and data structures
  params = rsa_corrs_setParams();
  modelRDMs = struct();

  %% import data
  % load behavioural data
  rsData = load([params.dir.behavDir 'rsData_scan_granada.mat']);
  fns = fieldnames(rsData);
  rsData = rsData.(fns{1});

  % load dissimilarity ratings
  dissimData = load([params.dir.dissimDir 'dissimData_exp1_granada_32subs.mat']);
  fns = fieldnames(dissimData);
  dissimData = dissimData.(fns{1});

  % load choie matrices
  results = load([params.dir.behavDir 'RTandCP_scan_granada.mat']);
  fns = fieldnames(results);
  cMats = results.(fns{1}).choicemat.orig;
  cMats_single = results.(fns{1}).choicemat.orig_singleRuns;

  %% generate models
  rdmSet = [];
  for (ii = 1:length(params.num.goodSubjects))
    subID = params.num.goodSubjects(ii);
    % - pixel space (using the images seen by subjects, averaged over exemplar RDMs)
    % obtain image labels, sort them
    for runID = 1:params.num.runs
      % extract trees for both contexts
      treesNorth = sort(rsData(subID).treeIDs(rsData(subID).data(:,1)==runID & rsData(subID).data(:,3)==1));
      treesSouth = sort(rsData(subID).treeIDs(rsData(subID).data(:,1)==runID & rsData(subID).data(:,3)==2));

      % group into 1x25,1x25 (for each context)
      % iterate through exemplar sets (should be two)
      for exmplID = 1:2
        treeSet(exmplID,:) = [treesNorth(exmplID:2:length(treesNorth)), treesSouth(exmplID:2:length(treesSouth))];
      end

      % load images and add background as well as contextual frame (same colour as during exp)
      % and reconstruct composite images (context + tree)
      northBG = imread([params.dir.bgDir 'north_garden.png']);
      southBG = imread([params.dir.bgDir 'south_garden.png']);
      northBG(northBG==178) = 150;
      southBG(southBG==178) = 150;
      for exmplID = 1:2
        respMat = [];
        % north garden
        for treeID = 1:50
          % load appropriate context
          if treeID <= 25
            img = northBG;
          else
            img = southBG;
          end
          % load tree
          tree = imread([params.dir.imageDir treeSet{exmplID,treeID} '.png']);
          tree(tree==0) = 150;
          % resize tree
          tree = imresize(tree,.3);
          % paste tree
          img(ceil(size(img,1)/2-size(tree,1)/2):floor(size(img,1)/2+size(tree,1)/2),ceil(size(img,2)/2-size(tree,2)/2):floor(size(img,2)/2+size(tree,2)/2),:) = tree;
          % vectorize tree and put in resp mat
          respMat(treeID,:) = img(:);
        end
        % for later RDM visualisation, save set of tree ctx composite images
        if (ii==1 & runID==1 & exmplID==1)
          treeMat = reshape(respMat,[50,242,250,3]);
          treeMat = cast(treeMat,'uint8');
          save('treeThumbnails.mat','treeMat');
        end
        respMat = cast(respMat,'uint8');
        rdmSet(ii,runID,exmplID,:,:) = squareform(pdist(respMat));
      end
    end
  end
  save('pixelRDMs_all.mat','rdmSet');
  % load('pixelRDMs_all.mat');
  rdms = squeeze(mean(mean(rdmSet,3),2));
  rdmSet = [];
  for subID = 1:size(rdms,1)
    rdmSet(ii,:,:) = helper_expandRDM(squeeze(rdms(subID,:,:)),params.num.runs);
  end
  modelRDMs(1).rdms = rdmSet;
  modelRDMs(1).name = 'pixel values';

  % - dissimilarity ratings (irrespective of context, obviously)
  rdmSet = [];
  for (ii = 1:length(params.num.goodSubjects))
    subID = params.num.goodSubjects(ii);
    for trialID = 1:max(dissimData(1).data(:,1))
      coords = dissimData(subID).data(dissimData(subID).data(:,1)==trialID,6:7);
      % expand for two contexts:
      coords = [coords;coords];
      dists = pdist(coords);
      dists = dists./max(dists);
      rdm   = squareform(dists);

      %expand RDM
      rdm = helper_expandRDM(rdm,params.num.runs);
      rdmSet(ii,trialID,:,:) = rdm;
    end
  end

  rdms = squeeze(mean(rdmSet,2));
  modelRDMs(2).rdms = rdms;
  modelRDMs(2).name = 'dissimilarity ratings';

%% MONOTONIC MODELS ------------------------------------------------------------
  % - relevant dimension  - 2D
  [b,l] = meshgrid([-2:2],[-2:2]);
  b = b(:);
  l = l(:);
  % respVect = [b,zeros(length(b),1);zeros(length(l),1),l];
  c1 = [cos(deg2rad(0));sin(deg2rad(0))];
  c2 = [cos(deg2rad(90));sin(deg2rad(90))];
  respVect = [[l,b]*c1*c1';[l,b]*c2*c2']; %l=north,b=south
  rdm = squareform(pdist(respVect));
  for (ii = 1:length(params.num.goodSubjects))
    subID = params.num.goodSubjects(ii);
    %expand RDM
    rdm2 = helper_expandRDM(rdm,params.num.runs);
    rdms(ii,:,:) = rdm2;
  end
  modelRDMs(3).rdms = rdms;
  modelRDMs(3).name = 'relevant dimension (2D to 1D)';

  % - irrelevant dimension - 2D
  [b,l] = meshgrid([-2:2],[-2:2]);
  b = b(:);
  l = l(:);
  % respVect = [b,zeros(length(b),1);zeros(length(l),1),l];
  c1 = [cos(deg2rad(0));sin(deg2rad(0))];
  c2 = [cos(deg2rad(90));sin(deg2rad(90))];
  respVect = [[l,b]*c2*c2';[l,b]*c1*c1']; %l=north,b=south
  rdm = squareform(pdist(respVect));
  for (ii = 1:length(params.num.goodSubjects))
    subID = params.num.goodSubjects(ii);
    %expand RDM
    rdm2 = helper_expandRDM(rdm,params.num.runs);
    rdms(ii,:,:) = rdm2;
  end
  modelRDMs(4).rdms = rdms;
  modelRDMs(4).name = 'irrelevant dimension (2D to 1D)';

  % - linear model - 2D
  for (ii = 1:length(params.num.goodSubjects))
    subID = params.num.goodSubjects(ii);
    boundID = rsData(subID).code(end);
    switch boundID
      case 1
        phi = 45;
      case 2
        phi = 225;
      case 3
        phi = 315;
      case 4
        phi = 135;
    end
    c = [cos(deg2rad(phi));sin(deg2rad(phi))];
    respVect = [[l,b]*c*c';[l,b]*c*c'];
    rdm = squareform(pdist(respVect));
    %expand RDM
    rdm = helper_expandRDM(rdm,params.num.runs);
    rdms(ii,:,:) = rdm;
  end
  modelRDMs(5).rdms = rdms;
  modelRDMs(5).name = 'linear model (2D to 1D)';

%% CATEGORICAL MODELS ----------------------------------------------------------
  % - factorised model - CHOICE -  contexts x feature
  for (ii = 1:length(params.num.goodSubjects))
    subID = params.num.goodSubjects(ii);
    boundID = rsData(subID).code(end);
    switch boundID
      case 1
        bv = [0 0 .5 1 1]-.5;
        lv = [0 0 .5 1 1]-.5;
      case 2
        bv = fliplr([0 0 .5 1 1]-.5);
        lv = fliplr([0 0 .5 1 1]-.5);
      case 3
        bv = [0 0 .5 1 1]-.5;
        lv = fliplr([0 0 .5 1 1]-.5);
      case 4
        bv = fliplr([0 0 .5 1 1]-.5);
        lv = [0 0 .5 1 1]-.5;
    end
    [b,l] = meshgrid(bv,lv);
    b = b(:);
    l = l(:);
    respVect = [l;b];
    rdm = squareform(pdist(respVect));
    %expand RDM
    rdm = helper_expandRDM(rdm,params.num.runs);
    rdms(ii,:,:) = rdm;
  end
  modelRDMs(6).rdms = rdms;
  modelRDMs(6).name = 'relevant dimension (binary)';

  % - linear model - CHOICE - 1D
  bv = [0 0 .5 1 1]-.5;
  lv = [0 0 .5 1 1]-.5;
  [b,l] = meshgrid(bv,lv);
  b = b(:);
  l = l(:);
  for (ii = 1:length(params.num.goodSubjects))
    subID = params.num.goodSubjects(ii);
    boundID = rsData(subID).code(end);
    switch boundID
      case 1
        phi = 45;
      case 2
        phi = 225;
      case 3
        phi = 315;
      case 4
        phi = 135;
    end
    c = [cos(deg2rad(phi));sin(deg2rad(phi))];
    respVect = [[l,b]*c;[l,b]*c];
    rdm = squareform(pdist(respVect));
    %expand RDM
    rdm = helper_expandRDM(rdm,params.num.runs);
    rdms(ii,:,:) = rdm;
  end
  modelRDMs(7).rdms = rdms;
  modelRDMs(7).name = 'linear model (binary)';

%% CONTEXT ---------------------------------------------------------------------
  % simple context encoding (A vs B)
  ctxNorth = zeros(25,1);
  ctxSouth = ones(25,1);
  respVect = [ctxNorth;ctxSouth];
  rdm = squareform(pdist(respVect));
  rdms = [];
  for (ii = 1:length(params.num.goodSubjects))
    subID = params.num.goodSubjects(ii);
    %expand RDM
    rdm2 = helper_expandRDM(rdm,params.num.runs);
    rdms(ii,:,:) = rdm2; % same division for all
  end
  modelRDMs(8).rdms = rdms;
  modelRDMs(8).name = 'garden (north vs south)';

%% EMPIRICAL CHOICES  ----------------------------------------------------------
  % participant's choices
  rdms = [];
  for (ii = 1:length(params.num.goodSubjects))
    subID = params.num.goodSubjects(ii);
    rn = squeeze(cMats_single.north(subID,1,:,:));
    rs = squeeze(cMats_single.south(subID,1,:,:));
    respVect = [rn(:);rs(:)];
    nConds = params.num.conditions;
    nRuns  =       params.num.runs;
    for runID = 2:params.num.runs
      rn = squeeze(cMats_single.north(subID,runID,:,:));
      rs = squeeze(cMats_single.south(subID,runID,:,:));
      rv = [rn(:);rs(:)];
      respVect = cat(1,respVect,rv);
    end
    rdm = squareform(pdist(respVect));
    for iiRun = 1:nConds:(nConds*nRuns)
      rdm(iiRun:iiRun+nConds-1,iiRun:iiRun+nConds-1) = NaN;;
    end
    rdms(ii,:,:) = rdm;
  end
  modelRDMs(9).rdms = rdms;
  modelRDMs(9).name = 'empirical choice patterns';

%% PERCEPTUAL FEATURE REPRESENTATIONS ------------------------------------------
  % only leafiness
  [b,l] = meshgrid([-2:2],[-2:2]);
  b = b(:);
  l = l(:);
  % respVect = [b,zeros(length(b),1);zeros(length(l),1),l];
  c1 = [cos(deg2rad(0));sin(deg2rad(0))];
  respVect = [[l,b]*c1*c1';[l,b]*c1*c1']; %l=north,b=south
  rdm = squareform(pdist(respVect));
  for (ii = 1:length(params.num.goodSubjects))
    subID = params.num.goodSubjects(ii);
    %expand RDM
    rdm2 = helper_expandRDM(rdm,params.num.runs);
    rdms(ii,:,:) = rdm2;
  end
  modelRDMs(10).rdms = rdms;
  modelRDMs(10).name = 'only leafiness';

  % only branchiness
  [b,l] = meshgrid([-2:2],[-2:2]);
  b = b(:);
  l = l(:);
  % respVect = [b,zeros(length(b),1);zeros(length(l),1),l];
  c1 = [cos(deg2rad(0));sin(deg2rad(0))];
  c2 = [cos(deg2rad(90));sin(deg2rad(90))];
  respVect = [[l,b]*c2*c2';[l,b]*c2*c2']; %l=north,b=south
  rdm = squareform(pdist(respVect));
  for (ii = 1:length(params.num.goodSubjects))
    subID = params.num.goodSubjects(ii);
    %expand RDM
    rdm2 = helper_expandRDM(rdm,params.num.runs);
    rdms(ii,:,:) = rdm2;
  end
  modelRDMs(11).rdms = rdms;
  modelRDMs(11).name = 'only branchiness';

  % grid
  [b,l] = meshgrid([-2:2],[-2:2]);
  b = b(:);
  l = l(:);
  respVect = [[l,b];[l,b]]; %l=north,b=south
  rdm = squareform(pdist(respVect));
  for (ii = 1:length(params.num.goodSubjects))
    subID = params.num.goodSubjects(ii);
    %expand RDM
    rdm2 = helper_expandRDM(rdm,params.num.runs);
    rdms(ii,:,:) = rdm2;
  end
  modelRDMs(12).rdms = rdms;
  modelRDMs(12).name = '"gridiness"';


  %% SINGLE TASK MODELS --------------------------------------------------------
    % grid - north only
    [b,l] = meshgrid([-2:2],[-2:2]);
    b = b(:);
    l = l(:);
    respVect = [[l,b];[l,b]]; %l=north,b=south
    rdm = squareform(pdist(respVect));
    rdm(26:end,26:end) = NaN;
    rdm(26:end,1:25) = NaN;
    rdm(1:25,26:end) = NaN;
    for (ii = 1:length(params.num.goodSubjects))
      subID = params.num.goodSubjects(ii);
      %expand RDM
      rdm2 = helper_expandRDM(rdm,params.num.runs);
      rdms(ii,:,:) = rdm2;
    end
    modelRDMs(13).rdms = rdms;
    modelRDMs(13).name = '"gridiness" - north task';

  % - relevant dimension  - 2D - north task
  [b,l] = meshgrid([-2:2],[-2:2]);
  b = b(:);
  l = l(:);
  % respVect = [b,zeros(length(b),1);zeros(length(l),1),l];
  c1 = [cos(deg2rad(0));sin(deg2rad(0))];
  c2 = [cos(deg2rad(90));sin(deg2rad(90))];
  respVect = [[l,b]*c1*c1';[l,b]*c2*c2']; %l=north,b=south
  rdm = squareform(pdist(respVect));
  rdm(26:end,26:end) = NaN;
  rdm(26:end,1:25) = NaN;
  rdm(1:25,26:end) = NaN;
  for (ii = 1:length(params.num.goodSubjects))
    subID = params.num.goodSubjects(ii);
    %expand RDM
    rdm2 = helper_expandRDM(rdm,params.num.runs);
    rdms(ii,:,:) = rdm2;
  end
  modelRDMs(14).rdms = rdms;
  modelRDMs(14).name = 'relevant dimension (2D to 1D) - north task';

  % - irrelevant dimension - 2D - north task
  [b,l] = meshgrid([-2:2],[-2:2]);
  b = b(:);
  l = l(:);
  % respVect = [b,zeros(length(b),1);zeros(length(l),1),l];
  c1 = [cos(deg2rad(0));sin(deg2rad(0))];
  c2 = [cos(deg2rad(90));sin(deg2rad(90))];
  respVect = [[l,b]*c2*c2';[l,b]*c1*c1']; %l=north,b=south
  rdm = squareform(pdist(respVect));
  rdm(26:end,26:end) = NaN;
  rdm(26:end,1:25) = NaN;
  rdm(1:25,26:end) = NaN;
  for (ii = 1:length(params.num.goodSubjects))
    subID = params.num.goodSubjects(ii);
    %expand RDM
    rdm2 = helper_expandRDM(rdm,params.num.runs);
    rdms(ii,:,:) = rdm2;
  end
  modelRDMs(15).rdms = rdms;
  modelRDMs(15).name = 'irrelevant dimension (2D to 1D) - north task';


  % grid - south task
  [b,l] = meshgrid([-2:2],[-2:2]);
  b = b(:);
  l = l(:);
  respVect = [[l,b];[l,b]]; %l=north,b=south
  rdm = squareform(pdist(respVect));
  rdm(1:25,1:25) = NaN;
  rdm(26:end,1:25) = NaN;
  rdm(1:25,26:end) = NaN;
  for (ii = 1:length(params.num.goodSubjects))
    subID = params.num.goodSubjects(ii);
    %expand RDM
    rdm2 = helper_expandRDM(rdm,params.num.runs);
    rdms(ii,:,:) = rdm2;
  end
  modelRDMs(16).rdms = rdms;
  modelRDMs(16).name = '"gridiness" - south task';

% - relevant dimension  - 2D - south task
[b,l] = meshgrid([-2:2],[-2:2]);
b = b(:);
l = l(:);
% respVect = [b,zeros(length(b),1);zeros(length(l),1),l];
c1 = [cos(deg2rad(0));sin(deg2rad(0))];
c2 = [cos(deg2rad(90));sin(deg2rad(90))];
respVect = [[l,b]*c1*c1';[l,b]*c2*c2']; %l=north,b=south
rdm = squareform(pdist(respVect));
rdm(1:25,1:25) = NaN;
rdm(26:end,1:25) = NaN;
rdm(1:25,26:end) = NaN;
for (ii = 1:length(params.num.goodSubjects))
  subID = params.num.goodSubjects(ii);
  %expand RDM
  rdm2 = helper_expandRDM(rdm,params.num.runs);
  rdms(ii,:,:) = rdm2;
end
modelRDMs(17).rdms = rdms;
modelRDMs(17).name = 'relevant dimension (2D to 1D) - south task';

% - irrelevant dimension - 2D - south task
[b,l] = meshgrid([-2:2],[-2:2]);
b = b(:);
l = l(:);
% respVect = [b,zeros(length(b),1);zeros(length(l),1),l];
c1 = [cos(deg2rad(0));sin(deg2rad(0))];
c2 = [cos(deg2rad(90));sin(deg2rad(90))];
respVect = [[l,b]*c2*c2';[l,b]*c1*c1']; %l=north,b=south
rdm = squareform(pdist(respVect));
rdm(1:25,1:25) = NaN;
rdm(26:end,1:25) = NaN;
rdm(1:25,26:end) = NaN;
for (ii = 1:length(params.num.goodSubjects))
  subID = params.num.goodSubjects(ii);
  %expand RDM
  rdm2 = helper_expandRDM(rdm,params.num.runs);
  rdms(ii,:,:) = rdm2;
end
modelRDMs(18).rdms = rdms;
modelRDMs(18).name = 'irrelevant dimension (2D to 1D) - south task';

%% EMPIRICAL PROJECTION AXIS (from psychophys model) ---------------------------
results = load([params.dir.behavDir 'psychophys_model_fits_ORIG_scan_granada.mat'])
phi_proj = deg2rad(results.results.phi - 90);
rdms = [];
for (ii = 1:length(params.num.goodSubjects))
    c1 = [cos(phi_proj(ii,1));sin(phi_proj(ii,1))];
    c2 = [cos(phi_proj(ii,2));sin(phi_proj(ii,2))];
    xy = [[b,l]*c1*c1';[b,l ]*c2*c2'];
    rdm = squareform(pdist(xy));
    %expand RDM
    rdm2 = helper_expandRDM(rdm,params.num.runs);
    rdms(ii,:,:) = rdm2;
end
modelRDMs(19).rdms = rdms;
modelRDMs(19).name = 'Empirical Projection Axes (2D to 1D)'

%% PARALLEL AXES ---------------------------------------------------------------
[b,l] = meshgrid([-2:2],[-2:2]);
b = b(:);
l = l(:);
c1 = [cos(deg2rad(0));sin(deg2rad(0))];
c2 = [cos(deg2rad(90));sin(deg2rad(90))];
xy = [[l,b]*c1*c1';[l,b ]*c2*c2'];
xy(26:end,:) = xy(26:end,:)*[cos(deg2rad(90)),-sin(deg2rad(90));sin(deg2rad(90)),cos(deg2rad(90))]; % rotate vector
xy(26:end,2) = xy(26:end,2) - 1; % make parallel
rdm = squareform(pdist(xy));

for (ii = 1:length(params.num.goodSubjects))
  subID = params.num.goodSubjects(ii);
  %expand RDM
  rdm2 = helper_expandRDM(rdm,params.num.runs);
  rdms(ii,:,:) = rdm2;
end
modelRDMs(20).rdms = rdms;
modelRDMs(20).name = 'parallel vectors, relevant dimensions (2D to 1D)';

%
%% PARALLEL PLANES (2D GRID for north and south task)
[b,l] = meshgrid([-2:2],[-2:2]);
 b = b(:);
 l = l(:);
 respVect = [[l,b,ones(length(b),1)];[l,b,2.*ones(length(b),1)]]; %l=north,b=south
 rdm = squareform(pdist(respVect));
 for (ii = 1:length(params.num.goodSubjects))
   subID = params.num.goodSubjects(ii);
   %expand RDM
   rdm2 = helper_expandRDM(rdm,params.num.runs);
   rdms(ii,:,:) = rdm2;
 end
 modelRDMs(21).rdms = rdms;
 modelRDMs(21).name = '"ctx * gridiness"';


%% ctx * relevant
[b,l] = meshgrid([-2:2],[-2:2]);
b = b(:);
l = l(:);
% respVect = [b,zeros(length(b),1);zeros(length(l),1),l];
c1 = [cos(deg2rad(0));sin(deg2rad(0))];
c2 = [cos(deg2rad(90));sin(deg2rad(90))];
respVect = [[l,b]*c1*c1',ones(length(b),1);[l,b]*c2*c2',2.*ones(length(b),1)]; %l=north,b=south
rdm = squareform(pdist(respVect));
for (ii = 1:length(params.num.goodSubjects))
  subID = params.num.goodSubjects(ii);
  %expand RDM
  rdm2 = helper_expandRDM(rdm,params.num.runs);
  rdms(ii,:,:) = rdm2;
end
modelRDMs(22).rdms = rdms;
modelRDMs(22).name = 'ctx * relevant dimension (2D to 1D)';

%% ctx * irrelevant
[b,l] = meshgrid([-2:2],[-2:2]);
b = b(:);
l = l(:);
% respVect = [b,zeros(length(b),1);zeros(length(l),1),l];
c1 = [cos(deg2rad(0));sin(deg2rad(0))];
c2 = [cos(deg2rad(90));sin(deg2rad(90))];
respVect = [[l,b]*c2*c2',ones(length(b),1);[l,b]*c1*c1',2.*ones(length(b),1)]; %l=north,b=south
rdm = squareform(pdist(respVect));
for (ii = 1:length(params.num.goodSubjects))
  subID = params.num.goodSubjects(ii);
  %expand RDM
  rdm2 = helper_expandRDM(rdm,params.num.runs);
  rdms(ii,:,:) = rdm2;
end
modelRDMs(23).rdms = rdms;
modelRDMs(23).name = 'ctx * irrelevant dimension (2D to 1D)';

%% ctx * only branchiness
% only branchiness
[b,l] = meshgrid([-2:2],[-2:2]);
b = b(:);
l = l(:);
% respVect = [b,zeros(length(b),1);zeros(length(l),1),l];
c1 = [cos(deg2rad(0));sin(deg2rad(0))];
c2 = [cos(deg2rad(90));sin(deg2rad(90))];
respVect = [[l,b]*c2*c2',ones(length(b),1);[l,b]*c2*c2',2.*ones(length(b),1)]; %l=north,b=south
rdm = squareform(pdist(respVect));
for (ii = 1:length(params.num.goodSubjects))
  subID = params.num.goodSubjects(ii);
  %expand RDM
  rdm2 = helper_expandRDM(rdm,params.num.runs);
  rdms(ii,:,:) = rdm2;
end
modelRDMs(24).rdms = rdms;
modelRDMs(24).name = 'ctx * only branchiness';
%% ctx * only leafiness
% only leafiness
[b,l] = meshgrid([-2:2],[-2:2]);
b = b(:);
l = l(:);
% respVect = [b,zeros(length(b),1);zeros(length(l),1),l];
c1 = [cos(deg2rad(0));sin(deg2rad(0))];
respVect = [[l,b]*c1*c1',ones(length(b),1);[l,b]*c1*c1',2.*ones(length(b),1)]; %l=north,b=south
rdm = squareform(pdist(respVect));
for (ii = 1:length(params.num.goodSubjects))
  subID = params.num.goodSubjects(ii);
  %expand RDM
  rdm2 = helper_expandRDM(rdm,params.num.runs);
  rdms(ii,:,:) = rdm2;
end
modelRDMs(25).rdms = rdms;
modelRDMs(25).name = 'ctx * only leafiness';


%% parallel planes, aligned with rewards
[b,l] = meshgrid([-2:2],[-2:2]);
b = b(:);
l = l(:);
c1 = [cos(deg2rad(0));sin(deg2rad(0))];
c2 = [cos(deg2rad(90));sin(deg2rad(90))];

for (ii = 1:length(params.num.goodSubjects))
  subID = params.num.goodSubjects(ii);
  boundID = rsData(subID).code(end);
  xy = [[l,b]*c1*c1';[l,b ]*c2*c2'];
  if boundID==3 || boundID == 4
      xy(26:end,:) = xy(26:end,:)*[cos(deg2rad(270)),-sin(deg2rad(270));sin(deg2rad(270)),cos(deg2rad(270))]; % rotate vector
  else
      xy(26:end,:) = xy(26:end,:)*[cos(deg2rad(90)),-sin(deg2rad(90));sin(deg2rad(90)),cos(deg2rad(90))]; % rotate vector
  end
  xy(26:end,2) = xy(26:end,2) - 1; % make parallel
  rdm = squareform(pdist(xy));
  %expand RDM
  rdm2 = helper_expandRDM(rdm,params.num.runs);
  rdms(ii,:,:) = rdm2;
end
modelRDMs(26).rdms = rdms;
modelRDMs(26).name = 'reward vectors, relevant dimensions (2D to 1D)';

%% overlapping planes, aligned with rewards
%TODO
end

function rdm = helper_expandRDM(rdm,nRuns)

  nConds = size(rdm,2);
  rdm = repmat(rdm,[nRuns,nRuns]);
  for iiRun = 1:nConds:(nConds*nRuns)
    rdm(iiRun:iiRun+nConds-1,iiRun:iiRun+nConds-1) = NaN;
  end

end
