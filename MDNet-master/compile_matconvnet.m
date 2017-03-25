%% COMPILE_MATCONVNET
%
% Compile MatConvNet
%
% Hyeonseob Nam, 2015 
%
compile4GPU = false;
run matconvnet/matlab/vl_setupnn ;
cd matconvnet;
if(compile4GPU)
    vl_compilenn('enableGpu', true, ...
               'cudaRoot', '/usr/local/cuda-6.5', ...
               'cudaMethod', 'nvcc');
else
    vl_compilenn;
end
cd ..;
