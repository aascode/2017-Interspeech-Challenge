%% Acquire Fisher Vector Encoding from Audio Features
%   This script acquires audio features from  form than 
%   output the fisher vector encoding as  features in xlsx
%   HGY, 02/18/17
% 
% Data format:
%     [features*D  idx*1]
% 
%  vlfeat download page: http://www.vlfeat.org/download.html
%  Remember to change path to your "vl_setup" & "vl_version.mexw64"

clear; clc;

%% Set Path  & Load Data
%  Path
OUT_ROOT = '../features/ComPare_2016/'
FEA_ROOT = '../features/ComPare_2016/';

%  load data
load([FEA_ROOT,'train4FV.mat'])
load([FEA_ROOT,'devel4FV.mat'])

% Initialize FV Toolbox
run('C:/Program Files/MATLAB/R2015b/toolbox/vlfeat-0.9.20/toolbox/vl_setup') 
run('C:/Program Files/MATLAB/R2015b/toolbox/vlfeat-0.9.20/toolbox/mex/mexw64/vl_version.mexw64') 


%% Parse Data
% train data
Idx_train = Data_train(:,end); 
GMM_input_train =  Data_train(:,1:end-1);

% test data
Idx_devel = Data_devel(:,end);
GMM_input_devel = Data_devel(:,1:end-1);
clear Data_train Data_devel

Sessions_train = unique(Idx_train);       % Total number of train data utterances
Sessions_devel = unique(Idx_devel);     % Total number of devel data utterances


%% Get Fisher Vectors
% GMM_Mixures=[2]; 
% GMM_Mixures = [4; 8; 16; 32]; 
% GMM_Mixures = [64; 128];
GMM_Mixures = [256; 512; 1024];

for index=1:length(GMM_Mixures)
    % Clear the output container
    FV_train = [];
    FV_devel = [];    
    
    % Build GMM model 
    Mixure=GMM_Mixures(index);
    fprintf(['Current mixure: ', int2str(Mixure),'\n'])
    [means, covariances, priors] = vl_gmm(GMM_input_train', Mixure , 'NumRepetitions', 5);
    
    % Use GMM model to get Fisher Encoding (for train)
    for ii = 0:1:length(Sessions_train)-1
        session = GMM_input_train(Idx_train==ii,:);
        FV_feature = [ii vl_fisher(session', means, covariances, priors, 'Improved')'];
        FV_train=[FV_train;FV_feature];
    end
    
    % Use GMM model to get Fisher Encoding (for devel)
    for ii = 0:1:length(Sessions_devel)-1
        session = GMM_input_devel(Idx_devel==ii,:);
        FV_feature = [ii vl_fisher(session', means, covariances, priors, 'Improved')'];
        FV_devel=[FV_devel;FV_feature];
    end
    
    % Output as .mat
    fileName=[OUT_ROOT,'FV_train_m',int2str(Mixure)];
    save([fileName,'.mat'],'FV_train')
    fileName=[OUT_ROOT,'FV_devel_m',int2str(Mixure)];
    save([fileName,'.mat'],'FV_devel')
end


