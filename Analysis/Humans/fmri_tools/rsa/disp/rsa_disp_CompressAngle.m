function xy = rsa_disp_CompressAngle(results)
    %% rsa_disp_CompressAngle(results)
    %
    % displays beta estimates for range of
    % feature angles of model RDM
    % lines indicate pure encoding of single dimensions
    % red circle highlights the group-level peak of the beta estimates
    %
    % Timo Flesch, 2019,
    % Human Information Processing Lab,
    % Experimental Psychology Department
    % University of Oxford

    figure(); set(gcf,'Color','w');
    xy = [];
    for ii = 1:size(results,1)
        [~,m] = max(squeeze(results(ii,:,:)),[],'all','linear');
        [xy(ii,2),xy(ii,1)] = ind2sub([size(results,2),size(results,3)],m);
    end

    %% kde plots
    [f1,xi1] = ksdensity(xy(:,1));
    [f2,xi2] = ksdensity(xy(:,2));
    f1 = rescale_values(f1, 1, 100/2);
    f2 = rescale_values(f2, 1, 180/2);
    idces = find(xi1<181 & xi1 > 1)
    xi1 = xi1(idces);
    f1 = f1(idces);
    idces = find(xi2<101 & xi2 > 1);
    xi2 = xi2(idces);
    f2 = f2(idces);

    % plot(xi1,f1);
    hold on;
    % plot(f2,xi2);
    fill([xi1,fliplr(xi1)],[ones(1,length(f1)),fliplr(f1)],[1,1,1].*.6,'LineStyle','none');
    hold on;
    fill([ones(1,length(f2)),fliplr(f2)],[xi2,fliplr(xi2)],[1,1,1].*.8,'LineStyle','none');
    %% single subject scatter
    ss = scatter(xy(:,1),xy(:,2));
    ss.MarkerFaceColor = [1,1,1].*.9;
    ss.MarkerEdgeColor = [0,0,0];
    hold on;
    set(gca,'XTick',1:45:181)
    xlim([1,181]);
    set(gca,'XTickLabel',[-90:45:90])
    set(gca,'YTick',1:25:101)
    ylim([1,101])
    set(gca,'YTickLabel',[0:0.25:1].*100)
    xlabel({'\bf rotation (°)'})
    ylabel({'\bf compression along irrelevant axis (%)'})
    axis square


    xy_max = floor(mean(xy,1))
    hold on
    plot([xy_max(1),xy_max(1)],get(gca,'YLim'),'LineStyle','--','Color',[1,0,0].*1);
    plot(get(gca,'XLim'),[xy_max(2),xy_max(2)],'LineStyle','--','Color',[1,0,0].*1);
    m = scatter(xy_max(1),xy_max(2),'ro','SizeData',80,'LineWidth',3)
    m.MarkerFaceColor = 'r';

    % title('Average ModelRDM Regression Coefficients','FontSize',16)
    grid on
    set(gcf,'Position',[660,555,497,371]);
    
end
