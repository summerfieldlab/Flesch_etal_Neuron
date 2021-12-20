function [rsd_switch,rsd_stay] = compute_split_rsData_switchstay(rsData)
    %% compute_split_rsData_switchstay(rsData)
    %
    % splits rsData into matrices with switch and stay trials 
    % 
    % Timo Flesch, 2020
    % Human Information Processing Lab 
    % University of Oxford 

    rsd_stay = rsData;
    rsd_switch = rsData;

    % iterate through subjects and split into switch/stay mats
    for ii = 1:length(rsData)
        stay_idces = rsData(ii).data(:,3) == circshift(rsData(ii).data(:,3),1);
        rsd_stay(ii).data = rsData(ii).data(stay_idces==1,:);
        rsd_switch(ii).data = rsData(ii).data(stay_idces==0,:);
    end 