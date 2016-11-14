close all;
addpath('testing/src');
addpath('../');
addpath('testing/util');
addpath('testing/util/ojwoodford-export_fig-5735e6d/');

%General Params
initIndex = 1;


%Files and Folders Params:
videoPath = 'video_files'; %folder with video sequence images(they should be numberred in ascending order).
heatmapsFolderName = 'heatmaps_video_files'; % folder with heatmapp mat files.

mkdir(heatmapsFolderName);
img_files = TrackerEvaluator.retrieveFileInformation(videoPath);
img_files = fullfile(videoPath, img_files);
[finalHeatMaps, prediction] = deal(cell(length(img_files), 1));

%% core: apply model to the images, to get heat maps and prediction coordinates
for k = initIndex:length(img_files)
    param = config();
    fprintf('Description of selected model: %s \n', param.model(param.modelID).description);
    info = imfinfo(img_files{k});
    rectangle = [1 1 info.Width info.Height];
    [heatMaps, prediction{k}] = applyModelShort(img_files{k}, param, rectangle);
    disp(k);
    currHeatMaps = heatMaps;
    finalHeatMap = currHeatMaps{param.model(param.modelID).stage};
    heatmapFileName = sprintf('finalHeatMap_%02d.mat', k);
    heatmapFilePath = fullfile(heatmapsFolderName, heatmapFileName);
    save(heatmapFilePath, 'finalHeatMap');  
end


