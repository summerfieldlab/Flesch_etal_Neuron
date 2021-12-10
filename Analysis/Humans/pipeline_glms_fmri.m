function pipeline_glms_fmri(repoDir)
    %
    % pipeline for all GLMs reported in paper 
    % note: this step might take several hours to complete

    %% import behavioural data 
    load([repoDir 'Data/Humans/part3_fmri/behav/scan/allData_scan.mat']);

    %% GLM1: switch vs stay trials     
    clearvars -except allData repoDir; close all; clc;
    % 1. specify regressors:
    glm_1_switchstay_genregressors(allData);
    % 2. estimate GLM:
    glm_1_switchstay_computeGLM();    
    % 3. compute 1st level contrasts:
    glm_1_switchstay_1stlvlcontrast();
    % 4. compute 2nd level contrasts:
    glm_1_switchstay_2ndlvlcontrast();


    %% GLM2: absolute distance to boundary     
    clearvars -except allData repoDir; close all; clc;
    % 1. specify regressors:
    glm_2_absdist_genregressors(allData);
    % 2. estimate GLM:
    glm_2_absdist_computeGLM();
    % 3. compute 1st level contrasts:
    glm_2_absdist_1stlvlcontrast();
    % 4. compute 2nd level contrasts:
    glm_2_absdist_2ndlvlcontrast();

    
    %% GLM3: signed distance to boundary     
    % clearvars -except allData repoDir; close all; clc;
    load([repoDir 'Data/Humans/part3_fmri/behav/scan/rsData_scan.mat']);
    % 1. specify regressors:
    glm_3_signeddist_genregressors(allData,rsData);
    % 2. estimate GLM:    
    glm_3_signeddist_computeGLM();
    % 3. compute 1st level contrasts:    
    glm_3_signeddist_1stlvlcontrast();
    % 4. compute 2nd level contrasts:
    glm_3_signeddist_2ndlvlcontrast();
    
    
    %% GLM4: RSA     
    clearvars -except allData repoDir; close all; clc;
    % 1. specify regressors:
    glm_4_rsa_genregressors(allData);
    % 2. estimate GLM:
    glm_4_rsa_computeGLM();  


    % %% GLM5: RSA switch vs stay    
    % clearvars -except allData repoDir; close all; clc;
    % % 1. specify regressors:
    % trial_ids = glm_5_rsa_switchstay_genregressors(allData);
    % % 2. estimate GLM:
    % glm_5_rsa_switchstay_computeGLM();
  
    
end