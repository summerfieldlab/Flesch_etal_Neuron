function results = compute_choicemodel_orig(choicemats,allData)
    %% compute_choicemodel()
    %
    % fits parametric choice model to behavioural data
    % to estimate lapse rates, sigmoid bias/slope and
    % deviation between learned and true decision boundary
    %
    % input: results.choicemat.orig (subfields north and south)
    % Timo Flesch, 2019,
    % Human Information Processing Lab,
    % Experimental Psychology Department
    % University of Oxford

    %% set params
    cat_bounds              =  [180, 90]; %optimal boundaries, factorised model
    init_vals   = [cat_bounds, 0, 20, 0]; % 180 for north, 90 for south task, zero offset and lapse, binarized choice probs
    constraints = struct();
    constraints.bound_north =  [0, 360];
    constraints.bound_south =   [0, 360];
    constraints.offset      =    [-1, 1];
    constraints.slope       =    [0, 20];
    constraints.lapserate   =   [0, 0.5];
    constraints_all = [constraints.bound_north;constraints.bound_south;constraints.offset;constraints.slope;constraints.lapserate];


    %% estimate model (single subject level)
    results = struct();
    [a,b] = meshgrid(-2:2,-2:2);
    x = [a(:),b(:)];
    % fid = 1;
    % figure();set(gcf,'Color','w');
    for ii = 1:size(choicemats.north,1)
        disp(['Processing subject ' num2str(ii)]);
        % set boundaries and constraints
        [cat_bounds, ~,~] = set_bounds(allData.subCodes(end,ii));
        % [cat_bounds, ~,~] = set_bounds(allData.ruleID(ii));
        init_vals   = [cat_bounds, 0, 20, 0];
        % process data
        cm_north = squeeze(choicemats.north(ii,:,:));
        cm_south = squeeze(choicemats.south(ii,:,:));
        y_true = [cm_north(:);cm_south(:)];
        [thetas,negLL] = fit_model(x,y_true,init_vals,constraints_all(:,1),constraints_all(:,2));

        bias= [compute_boundaryBias(thetas(1),cat_bounds(1),'north'),...
        compute_boundaryBias(thetas(2),cat_bounds(2),'south')];
        % five parameter model (phi_A,phi_B,slope,offset,lapse)
        results.slope(ii,:)  = thetas(4);
        results.offset(ii,:) = thetas(3);
        results.lapse(ii,:)  = thetas(5);
        results.bias(ii,:)   = squeeze(mean(bias));
        results.phi(ii,:)    = squeeze(thetas(1:2));
        results.nll(ii,:)    = negLL;
        results.bic(ii,:)    = compute_BIC(-negLL,length(thetas),50); % goodness of fit
        results.gt_phi(ii,:) = cat_bounds;

        % y_hat = choicemodel(x,thetas);
        % y_north = reshape(y_hat(1:25),[5,5]);
        % y_south = reshape(y_hat(26:end),[5,5]);
        % subplot(8,16,fid);
        % imagesc(cm_north);
        % axis square;
        % title(num2str(allData.ruleID(ii)));
        % subplot(8,16,fid+16);
        % imagesc(cm_south);
        % axis square;
        % subplot(8,16,fid+32);
        % imagesc(y_north);
        % subplot(8,16,fid+48);
        % imagesc(y_south);
        %
        % if fid==16
        %     fid = 64;
        % end
        % fid = fid+1;
    end

end



function y_hat = choicemodel(X,theta)
    %
    % a parametric model for single subject choices
    X1 = scalarproj(X,theta(1));
    X2 = scalarproj(X,theta(2));
    y_hat = transducer([X1;X2],theta(3),theta(4),theta(5));

    function y = scalarproj(x,phi)
        phi_bound =                  deg2rad(phi);
        phi_ort   =         phi_bound-deg2rad(90);
        y         = x*[cos(phi_ort);sin(phi_ort)];
    end

    function y = transducer(x,offset,slope,lapse)
        y = lapse + (1-lapse*2)./(1+exp(-slope*(x-offset)));
    end
end


function [betas,loss] = fit_model(x,y_true,init_vals,lb,ub)
    %
    % minimises objective function
    % returns best fitting parameters

    % define objective function
    loss = @(init_vals)-sum(log(1-abs(y_true(:)-choicemodel(x,init_vals))+1e-10));


    % fit model
    [betas,loss] = fmincon(loss,init_vals,[],[],[],[],lb,ub,[],optimoptions('fmincon','Display','off'));
end


function boundaryBias = compute_boundaryBias(estimatedAngle,catBound,task)
	% we interpret a higher positive bias as stronger tendency towards a combined representations
	% hence the sign flip for north (combined would be 90, but optimal is 180 deg)
	switch task
		case 'north'
			% boundaryBias = abs(rad2deg(circ_dist(deg2rad(estimatedAngle),deg2rad(catBound))));
			boundaryBias = -(rad2deg(circ_dist(deg2rad(estimatedAngle),deg2rad(catBound))));
		case 'south'
			boundaryBias = rad2deg(circ_dist(deg2rad(estimatedAngle),deg2rad(catBound)));
			% boundaryBias = abs(rad2deg(circ_dist(deg2rad(estimatedAngle),deg2rad(catBound))));
	end
end

function [bounds, lim_north, lim_south] = set_bounds(code)

    switch code
    case 1
        bounds = [180, 90];
        lim_north = [90,270];
        lim_south = [0,180];
    case 2
        bounds = [0, 270];
        lim_north = [-90,90];
        lim_south = [180,360];
    case 3
        bounds = [0, 90];
        lim_north = [-90,270];
        lim_south = [90,180];
    case 4
        bounds = [180, 270];
        lim_north = [90,270];
        lim_south = [180,360];
    end
end
