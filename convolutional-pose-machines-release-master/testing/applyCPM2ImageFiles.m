%% Demo code of "Convolutional Pose Machines", 
% Shih-En Wei, Varun Ramakrishna, Takeo Kanade, Yaser Sheikh
% In CVPR 2016
% Please contact Shih-En Wei at shihenw@cmu.edu for any problems or questions
%%
close all;
addpath('src');
addpath('../../');
addpath('util');
addpath('util/ojwoodford-export_fig-5735e6d/');
initIndex = 3;
kinectDateVector = [29 7 16 18 2];
prePath = '/home/felix/BGU_Computer_Vision_thesis/Codes/convolutional-pose-machines-release-master/';
addpath(prePath);

[~, resultFileName, videoPath] = TrackerEvaluator.createFileNamesFromDateVector(kinectDateVector, false);
videoPath = [prePath videoPath];
resultFilePath = [prePath 'CPMbest' resultFileName];
[~, pureFileName, ~] = fileparts(resultFilePath);
C = strsplit(pureFileName,'results');
dateSpecialStr = C{2};
newFolderName = sprintf('heatMapDeff_%s', dateSpecialStr);
mkdir([prePath newFolderName]);
newFolderName = [prePath newFolderName];
[img_files, filenums] = TrackerEvaluator.retrieveFileInformation(videoPath);
img_files = fullfile(videoPath, img_files);
[finalHeatMaps, prediction] = deal(cell(length(img_files), 1));

%% core: apply model to the images, to get heat maps and prediction coordinates
for k = initIndex:length(img_files)
    param = config();
    fprintf('Description of selected model: %s \n', param.model(param.modelID).description);
    info = imfinfo(img_files{k});
    rectangle = [1 1 info.Width info.Height];
    [heatMaps, prediction{k}] = applyModelShort(img_files{k}, param, rectangle);
%     myVisualize(img_files{k}, heatMaps, prediction{k}, param, rectangle);
    disp(k);
    assert(false);
    currHeatMaps = heatMaps;
    finalHeatMap = currHeatMaps{param.model(param.modelID).stage};
    heatmapFileName = sprintf('finalHeatMap_%02d.mat', k);
    heatmapFilePath = fullfile(newFolderName, heatmapFileName);
    save(heatmapFilePath, 'finalHeatMap');
    
end
save(resultFilePath, 'prediction', 'k');

