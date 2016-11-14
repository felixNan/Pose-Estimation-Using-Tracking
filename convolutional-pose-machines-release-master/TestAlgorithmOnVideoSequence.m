clear ;
close all;
clc;


%Files and Folders Params:
heatMapfolderName = 'heatMapDeff_25_05_16_00_53'; % folder with heatmapp mat files.
videoPath = 'kinectVideoFrames_25_05_16_00_53'; %folder with video sequence images(they should be numberred in ascending order).
outputVideoFileName = 'outputVideoTest.avi'; %output video of per frame estimator.
outputVideoFileNameAlg = 'outputVideoAlgTest.avi'; %output video of algorithm with tracking enhancement.

%General Params andPose Alg Params:
relevantJoints = 8;
doRunAlg = true;
poseAlgParams.relativeRadius = 0.05;
poseAlgParams.peakProbabiltyThresh = 0.2;
poseAlgParams.probabiltyInRadiusThresh = 0.18;
poseAlgParams.avergingWindowSize = 15;
poseAlgParams.confidenceThresh = 0.7;
poseAlgParams.doUseMDNet = true;
poseAlgParams.minFrames2Track =  0;
poseAlgParams.posSize4Tracker = 0.05;
     
TrackerEvaluator.prepareMDNet4Work();
startIndex = 3;        
        
%% Extract heatMap and prediction from folders.

heatMapDirInfo = dir([heatMapfolderName '/*.mat']);
filesCellArray = {heatMapDirInfo.name}';
numOfFiles = length(filesCellArray);    
numIncrease = startIndex -1;
[prediction, heatMapCellArray] = deal(cell(numOfFiles + numIncrease, 1));

for k= startIndex :numOfFiles + numIncrease
	heatMapFile = fullfile(heatMapfolderName, sprintf('finalHeatMap_%02d.mat', k));
	load(heatMapFile);
	heatMapCellArray{k} = finalHeatMap;
	np = size(finalHeatMap, 3) - 1;
	for m = 1:np
		[x, y] = jointsAlgManager.findHeatMapMaximum(finalHeatMap(:,:,m));
		prediction{k} (m,:)=  [x, y];
	end
	fprintf('Recovered prediction No %d\n', k);
end



if (doRunAlg)
	[algResults, isConfidentVectors, trackingSpans] = jointsAlgManager.poseEstimationUsingTrackingAlg(poseAlgParams, heatMapCellArray, relevantJoints, videoPath, false, startIndex);
end




[isConfidentVector, peakProbabilities, probabilitiesInRadius] = deal(cell(relevantJoints, 1));
for iJoint = 1: relevantJoints
	[isConfidentVector{iJoint}, peakProbabilities{iJoint}, probabilitiesInRadius{iJoint}] = jointsAlgManager.obtainConfidence4Video(heatMapCellArray, poseAlgParams, iJoint);
	isConfidentVector{iJoint} = isConfidentVector{iJoint}(startIndex:end);
	peakProbabilities{iJoint} = peakProbabilities{iJoint}(startIndex:end);
	probabilitiesInRadius{iJoint} = probabilitiesInRadius{iJoint}(startIndex:end);
	fprintf('Obtained confidence params for joint number %d\n', iJoint);
end


%% Show predicted joints
jointImages = jointsAlgManager.drawUpperBodyJoints(prediction, startIndex, videoPath, 'red', true);

if (doRunAlg)
	jointImagesAlg = jointsAlgManager.drawUpperBodyJoints(algResults, startIndex, videoPath, 'red', true);
end

jointsAlgManager.createVideoFromJointImages(jointImages, outputVideoFileName, startIndex, 10);

if (doRunAlg)
	jointsAlgManager.createVideoFromJointImages(jointImagesAlg, outputVideoFileNameAlg, startIndex, 10);
end

