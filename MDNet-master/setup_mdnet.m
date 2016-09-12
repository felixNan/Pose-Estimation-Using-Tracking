%% SETUP_MDNET
%
% Setup the environment for running MDNet.
%
% Hyeonseob Nam, 2015 
%

if(isempty(gcp('nocreate')))
    parpool;
end
compile_matconvnet;
run matconvnet/matlab/vl_setupnn ;

addpath('pretraining');
addpath('tracking');
addpath('utils');
