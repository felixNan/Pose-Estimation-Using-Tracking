matfilename ='CPMbestresults23_05_16_02_02.mat';
[~, pureFileName, ~] = fileparts(matfilename);
C = strsplit(pureFileName,'results');
dateSpecialStr = C{2};
newFolderName = sprintf('heatMapDeff_%s', dateSpecialStr);
mkdir(newFolderName);
load(matfilename,'finalHeatMaps');
for k = 1: length(finalHeatMaps)
    finalHeatMap = finalHeatMaps{k};
    heatmapFileName = sprintf('finalHeatMap_%02d.mat', k);
    heatmapFilePath = fullfile(newFolderName, heatmapFileName);
    save(heatmapFilePath, 'finalHeatMap');
end
