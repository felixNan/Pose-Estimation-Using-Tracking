clear ;
close all;
clc;


%dateVectors = [23 5 16 2 2; 25 5 16 0 53; 25 5 16 0 59; 29 7 16 18 0; 29 7 16 18 2; 30 5 16 0 38];
%avgWindowSizes = [5; 10; 15; 20];
dateVectors = [25 5 16 0 53];
avgWindowSizes = 10;

for iWindowSize = 1:length(avgWindowSizes)
    for k = 1: size(dateVectors, 1)
        dateVector = dateVectors(k,:);
        relevantJoints = 8;
        doRunAlg = true;
        TrackerEvaluator.prepareMDNet4Work();
        
        dateStringFolder = sprintf('_%02d', dateVector);
        dateStringMat = sprintf('%02d_', dateVector);
        dateStringMat = dateStringMat(1:end-1);
        heatMapfolderName = ['heatMapDeff' dateStringFolder] ;
        videoPath = ['kinectVideoFrames'  dateStringFolder filesep];
        jointMatPath = ['kinectJointsData'  dateStringMat '.mat'];
        outputVideoFileName = [ 'outputVideo' dateStringMat '.avi'];
        windowSizeString = sprintf('avfWindowSize_%d',avgWindowSizes(iWindowSize));
        outputVideoFileNameAlg = [ 'outputVideoAlg' dateStringMat  windowSizeString '.avi'];
        metricsMatFile = ['metricResults' dateStringMat windowSizeString '.mat'];
        
        %% Extract heatMap and prediction from folders.
        heatMapDirInfo = dir([heatMapfolderName '/*.mat']);
        filesCellArray = {heatMapDirInfo.name}';
        numOfFiles = length(filesCellArray);
        startIndex = 3;
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
        
        % trackingDistances = 2:8;
        % numOfTrackingDistances = length(trackingDistances);
        % [metricMeans,metricMaxes, avgNumBigErrors] = deal(zeros(numOfTrackingDistances, 1));
        %
        % for q = 1:numOfTrackingDistances
        
        if (doRunAlg)
            poseAlgParams.relativeRadius = 0.05;
            poseAlgParams.peakProbabiltyThresh = 0.35;
            poseAlgParams.probabiltyInRadiusThresh = 0.27;
            poseAlgParams.avergingWindowSize = 15;
            poseAlgParams.confidenceThresh = 0.7;
            poseAlgParams.doUseMDNet = true;
            poseAlgParams.minFrames2Track =  0;
            [algResults, isConfidentVectors, trackingSpans] = jointsAlgManager.poseEstimationUsingTrackingAlg(poseAlgParams, heatMapCellArray, 8, videoPath, false, 3);
        end
        
        
        %% Compute metric and joint confidence
        load(jointMatPath);
        cpmJointMetric = jointsAlgManager.computeCPMMetricRelevantToKinectJoints(videoPath, startIndex, prediction, jointLists);
        
        if (doRunAlg)
            cpmJointMetricTracking = jointsAlgManager.computeCPMMetricRelevantToKinectJoints(videoPath, startIndex, algResults, jointLists);
        end
        % metricMeans(q) = mean(cpmJointMetricTracking);
        % metricMaxes(q) = max(cpmJointMetricTracking);
        % avgNumBigErrors(q) = nnz(cpmJointMetricTracking> 60)/length(cpmJointMetricTracking);
        % end
        % videoMetricMean = mean(cpmJointMetric);
        % videoMetricMax = max(cpmJointMetric);
        % videoAvgNumOfBigErros = nnz(cpmJointMetric > 60)/length(cpmJointMetric);
        % assert(false);%temporary measure
        
        [isConfidentVector, peakProbabilities, probabilitiesInRadius] = deal(cell(relevantJoints, 1));
        for iJoint = 1: relevantJoints
            [isConfidentVector{iJoint}, peakProbabilities{iJoint}, probabilitiesInRadius{iJoint}] = jointsAlgManager.obtainConfidence4Video(heatMapCellArray, 0.05, 0.4, 0.3, iJoint);
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
        
        save(metricsMatFile, 'jointImagesAlg', 'jointImages',  'cpmJointMetricTracking', 'cpmJointMetric'); %saving separate mat-file for each date vector
        
    end
end
