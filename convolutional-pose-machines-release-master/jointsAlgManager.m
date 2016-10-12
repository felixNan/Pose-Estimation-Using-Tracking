classdef jointsAlgManager < handle
    %% Public Static Methods
    methods (Static)
        
        function imJoints = plotAllJoints(I, jointsMat, jointColors)
            imJoints = insertMarker(I, jointsMat', 'x', 'Color', jointColors, 'Size', 10);
        end
        
        function [isConfident, peakProbabilty, probabiltyInARadius] = checkIfConfidentJoint(jointHeatmap, relativeRadius, peakProbabiltyThresh, probabiltyInRadiusThresh)
            
            absRadius = relativeRadius * min(size(jointHeatmap));
            
            [columnIndex, rowIndex] = jointsAlgManager.findHeatMapMaximum(jointHeatmap);
            
            logicalEmptyImage = false(size(jointHeatmap));
            logicalCircleImage = insertShape(double(logicalEmptyImage), 'FilledCircle', [columnIndex rowIndex absRadius], 'LineWidth', 1);
            logicalCircleImage = rgb2gray(logicalCircleImage) > 0;
            heatmapInACircle = logicalCircleImage .* jointHeatmap;
            probabiltyInARadius = sum(heatmapInACircle(:)) / sum(jointHeatmap(:));
            peakProbabilty = jointHeatmap(rowIndex ,columnIndex);
            isConfident = peakProbabilty > peakProbabiltyThresh & probabiltyInARadius > probabiltyInRadiusThresh ;
            
        end
        
        function [isConfidentVector, peakProbabilities, probabilitiesInRadius] = obtainConfidence4Video(heatmapCellArray, confidenceAlgParams, jointNumber)
            
            relativeRadius = confidenceAlgParams.relativeRadius;
			peakProbabiltyThresh = confidenceAlgParams.peakProbabiltyThresh;
			probabiltyInRadiusThresh = confidenceAlgParams.probabiltyInRadiusThresh;
            
            numOfFrames = length(heatmapCellArray);
            [isConfidentVector, peakProbabilities, probabilitiesInRadius] = deal(zeros(numOfFrames, 1));
            for k = 3:numOfFrames
                heatmapImages = heatmapCellArray{k};
                heatmapImage = heatmapImages(: ,: ,jointNumber);
                [isConfidentVector(k), peakProbabilities(k), probabilitiesInRadius(k)] = jointsAlgManager.checkIfConfidentJoint(heatmapImage, relativeRadius, peakProbabiltyThresh, probabiltyInRadiusThresh);
            end
        end
        
        function [algResults, isConfidentVectors, trackingSpans] = poseEstimationUsingTrackingAlg(poseAlgParams, heatmapImages, numberOfJoints, videoPath, isBlind, startIndx)
            %gets results for all joints.
            imgFiles = TrackerEvaluator.retrieveFileInformation(videoPath);
            numOfImages = length(imgFiles);
            [algResults, trackingSpans]  = deal(cell(numberOfJoints, 1));
            isConfidentVectors = zeros(numOfImages, numberOfJoints);
            
            for m = 1 : numberOfJoints
                if (isBlind)
                    [algResults{m}, isConfidentVectors(:, m), trackingSpans{m}]  = jointsAlgManager.estimateSingleJointBlindly(heatmapImages, imgFiles, poseAlgParams, m, videoPath, startIndx);
                else
                    [algResults{m}, isConfidentVectors(:, m), trackingSpans{m}]  = jointsAlgManager.estimateSingleJointBetweenConfidentFrames(heatmapImages, imgFiles, poseAlgParams, m, videoPath, startIndx);
                end
                fprintf('finished tracking Joint %d\n', m);
            end
            
            %converting algResults to CPM prediction format
            algResults = jointsAlgManager.convertAlgoJointsToUpperBodyFormat(algResults, numberOfJoints);
        end
        
        
        function [isConfidentVector, peakProbabilities, probabilitiesInRadius] = obtainConfidence4VideoConv(heatmapCellArray, confidenceAlgParams, jointNumber, startIndx)
      
			relativeRadius = confidenceAlgParams.relativeRadius;
			peakProbabiltyThresh = confidenceAlgParams.peakProbabiltyThresh;
			probabiltyInRadiusThresh = confidenceAlgParams.probabiltyInRadiusThresh;
			avergingWindowSize = confidenceAlgParams.avergingWindowSize;
			confidenceThresh = confidenceAlgParams.confidenceThresh;
      
			numOfFrames = length(heatmapCellArray);
			avgWindow = ones(avergingWindowSize, 1)/avergingWindowSize;
			[isConfidentVector, peakProbabilities, probabilitiesInRadius] = deal(zeros(numOfFrames, 1));
			for k = startIndx:numOfFrames
				heatmapImages = heatmapCellArray{k};
				heatmapImage = heatmapImages(: ,: ,jointNumber);
                [~ , peakProbabilities(k), probabilitiesInRadius(k)] = jointsAlgManager.checkIfConfidentJoint(heatmapImage, relativeRadius, peakProbabiltyThresh, probabiltyInRadiusThresh);
			end
      
			peakProbabilitiesAverages = conv(peakProbabilities, avgWindow, 'same');
			probabilitiesInRadiusAverages =  conv(probabilitiesInRadius, avgWindow,'same');
			isConfidentVector(startIndx : avergingWindowSize + startIndx) = 1;
			logIndicesOfConfidentFrames = peakProbabilities > confidenceThresh * peakProbabilitiesAverages & probabilitiesInRadius > confidenceThresh * probabilitiesInRadiusAverages;
			isConfidentVector(avergingWindowSize + startIndx + 1:end) =  logIndicesOfConfidentFrames(avergingWindowSize + startIndx + 1:end); %first part is thresh because of the convolution.
      
		end
        
        
        function [perJointAlgResults, isConfidentVector, trackingSpanLength] = estimateSingleJointBlindly(heatmapImages ,img_files ,poseAlgParams ,jointNumber, videoPath, startIndx)
            
            relativeRadius = poseAlgParams.relativeRadius;
            doUseMDNet = poseAlgParams.doUseMDNet;
            minFrames2Track = poseAlgParams.minFrames2Track;
            
            trackingSpanLength = [];
            isConfidentVector = jointsAlgManager.obtainConfidence4Video(heatmapImages, poseAlgParams, jointNumber);
            if (doUseMDNet)
                net = fullfile('/home/felix/BGU_Computer_Vision_thesis/Codes/Pose-Estimation-Using-Tracking/MDNet-master','models','mdnet_vot-otb_cpu.mat');
            end
            %gets result for a single joint.
            numOfImages = length(img_files);
            perJointAlgResults = cell(numOfImages, 1);
            nextIndex = startIndx;
            while (nextIndex <= numOfImages)
                %tracking between confident indices(condfident pose
                %estimations), for minFrames2Track frames at least.
                %Tracking end one frame before a confident frame.
                currentIndex = nextIndex ;
                nextIndex = min(numOfImages + 1, currentIndex + minFrames2Track);
                trackingSpan = currentIndex : nextIndex - 1;
                trackingSpanLength = [trackingSpanLength ; length(trackingSpan)]; % not really needed, used just for gathering info on actual number of tracked images.
                currHeatMapImage = heatmapImages{currentIndex};
                initPos =  jointsAlgManager.getJointLocationFromHeatMap(currHeatMapImage, jointNumber) ;
                posWidth = 2 * relativeRadius * min(size(currHeatMapImage(:, :, jointNumber)));
                posHeight = posWidth;
                relevantImgFiles = img_files(trackingSpan);
                if (length(relevantImgFiles) > 1)
                    if (doUseMDNet)
                        trackingResults = TrackerEvaluator.performSimpleMDNetTracking(videoPath, initPos, posWidth, posHeight, net, relevantImgFiles, false);
                    else
                        trackingResults = TrackerEvaluator.performSimpleKCFTracking(videoPath ,initPos, posWidth, posHeight, relevantImgFiles, false);
                    end
                    numOfTrackedFrames = size(trackingResults, 1);
                    perJointAlgResults(trackingSpan, :) = mat2cell(trackingResults, ones(1, numOfTrackedFrames));
                end
            end
        end
        
        function [perJointAlgResults, isConfidentVector, trackingSpanLength] = estimateSingleJointBetweenConfidentFrames(heatmapImages, img_files, poseAlgParams, jointNumber, videoPath, startIndx)
            
            relativeRadius = poseAlgParams.relativeRadius;
            doUseMDNet = poseAlgParams.doUseMDNet;
            minFrames2Track = poseAlgParams.minFrames2Track;
            
            trackingSpanLength = [];
            isConfidentVector = jointsAlgManager.obtainConfidence4Video(heatmapImages, poseAlgParams, jointNumber);
            if (doUseMDNet)
                net = fullfile('/home/felix/BGU_Computer_Vision_thesis/Codes/Pose-Estimation-Using-Tracking/MDNet-master','models','mdnet_vot-otb_cpu.mat');
            end
            
            numOfImages = length(img_files);
            [confidenceIndices, perJointAlgResults] = jointsAlgManager.obtainAlgResultsForConfidenceImages(heatmapImages, isConfidentVector, jointNumber, startIndx);
            
            nextConfidentIndex = startIndx;
            while (nextConfidentIndex <= numOfImages)
                %tracking between confident indices(condfident pose
                %estimations), for minFrames2Track frames at least.
                %Tracking end one frame before a confident frame.
                currentConfidentIndex = nextConfidentIndex ;
                indx = find(confidenceIndices > currentConfidentIndex + minFrames2Track, 1, 'first');
                if isempty(indx)
                    nextConfidentIndex = numOfImages + 1;
                else
                    nextConfidentIndex  = confidenceIndices(indx);
                end
                trackingSpan = currentConfidentIndex: nextConfidentIndex - 1;
                trackingSpanLength = [trackingSpanLength ; length(trackingSpan)]; % not really needed, used just for gathering info on actual number of tracked images.
                currHeatMapImage = heatmapImages{currentConfidentIndex};
                initPos =  jointsAlgManager.getJointLocationFromHeatMap(currHeatMapImage, jointNumber) ;
                posWidth = 2 * relativeRadius * min(size(currHeatMapImage(:, :, jointNumber)));
                posHeight = posWidth;
                relevantImgFiles = img_files(trackingSpan);
                if (length(relevantImgFiles) > 1)
                    fprintf('Tracking joint number %d, from frame number %02d\n', jointNumber, currentConfidentIndex);
                    if (doUseMDNet)
                        trackingResults = TrackerEvaluator.performSimpleMDNetTracking(videoPath, initPos, posWidth, posHeight, net, relevantImgFiles, false);
                    else
                        trackingResults = TrackerEvaluator.performSimpleKCFTracking(videoPath ,initPos, posWidth, posHeight, relevantImgFiles, false);
                    end
                    numOfTrackedFrames = size(trackingResults, 1);
                    perJointAlgResults(trackingSpan, :) = mat2cell(trackingResults, ones(1, numOfTrackedFrames));
                end
            end
        end
        
        function jointImages = drawUpperBodyJoints(jointsFromAlgorithmCell, startIndex, videoPath, color, isFlowing)
            imageFileCell = TrackerEvaluator.retrieveFileInformation(videoPath);
            imageFileCell = fullfile(videoPath, imageFileCell);
            numOfImages = length(imageFileCell);
            jointImages = cell(numOfImages, 1);
            for k = startIndex:numOfImages
                cpmJointsInKinectFormat = jointsAlgManager.convertCPMJoints2KinectFormat(jointsFromAlgorithmCell{k});
                imageMatrix = imread(imageFileCell{k});
                jointImages{k} = jointsAlgManager.insertJointLinesToImage(imageMatrix ,cpmJointsInKinectFormat, color);
                imshow(jointImages{k}, []);
                if (isFlowing)
                    title(sprintf('displaying frame number %d', k));
                    pause(0.05);
                else
                    pause;
                    endstartIndx
                end
                
            end
        end
        function createVideoFromJointImages(jointImages, outputVideoFileName, startIndex, FrameRate)
            outputVideo = VideoWriter(outputVideoFileName);
            outputVideo.FrameRate = FrameRate;
            position =  [1 50];
            open(outputVideo);
            for iImage = startIndex:length(jointImages)
                overlayStr = sprintf('frame number %d', iImage - startIndex -1);
                img =insertText(jointImages{iImage}, position, overlayStr, 'AnchorPoint', 'LeftBottom');
                writeVideo(outputVideo, img);
            end
            close(outputVideo);
        end
        
        function im = insertJointLinesToImage(I, jointLocations, color)
            connMap = [6 4; 7 5; 4 2; 5 3];
            startPoints = connMap(:,1);
            endPoints = connMap(:,2);
            LineMat = [jointLocations(startPoints, :) jointLocations(endPoints, :)];
            im = insertShape(I, 'Line', LineMat, 'LineWidth', 3, 'Color', color);
            im = insertMarker(im, jointLocations, 'marker' ,'x', 'Color', 'yellow');
        end
        
        
        function cpmJointMetric = computeCPMMetricRelevantToKinectJoints(videoPath, startIndex, prediction, jointLists)
            jointListsKinect = kinectHandler.getOnlyRelevantKinectJoints(videoPath, startIndex, jointLists);
            cpmJointListInKinectFormat = cellfun(@jointsAlgManager.convertCPMJoints2KinectFormat, prediction(startIndex:end), 'UniformOutput', false);
            numOfEstimatedFrames = length(cpmJointListInKinectFormat);
            cpmJointMetric = zeros(numOfEstimatedFrames, 1);
            for  k =1: numOfEstimatedFrames
                jointListsKinect{k} = kinectHandler.blobJointsInKinectCoords(jointListsKinect{k}, [256 256], [480 640]);
                cpmJointMetric(k) = jointsAlgManager.computeMaxJointDistance(cpmJointListInKinectFormat{k}, jointListsKinect{k});
            end
        end
        
        
        function cpmJointsInKinectFormat = convertCPMJoints2KinectFormat(jointLocations)
            jointIndexArray = [1 5 8 4 7 3 6];
            cpmJointsInKinectFormat = jointLocations(jointIndexArray ,:);
            cpmJointsInKinectFormat(1,:) = (jointLocations(2,:) + jointLocations(1,:)) * 0.5; %head joints should be in the middle between CPM head and neck.
        end
        
        function algoJointsInUpperBodyFormat = convertAlgoJointsToUpperBodyFormat(algResults, numOfRelevantJoints)
            numOfFrames = length(algResults{1});
            algoJointsInUpperBodyFormat = cell(numOfFrames, 1);
            for iJoint = 1: numOfRelevantJoints
                algResultsPerJoint = algResults{iJoint};
                for iFrame =1 : numOfFrames
                    if(~isempty(algResultsPerJoint{iFrame}))
                        algoJointsInUpperBodyFormat{iFrame}(iJoint, :) = algResultsPerJoint{iFrame};
                    end
                end
            end
        end
        
        function [x, y] = findHeatMapMaximum(heatMapImage)
            [~, linearIndex] = max(heatMapImage(:));
            [y, x] = ind2sub(size(heatMapImage), linearIndex);
        end
        
    end
    %% Private
    methods (Static, Access = private)
        
        
        function pos  = getJointLocationFromHeatMap(HeatMapImage, jointNumber)
            jointHeatMap = HeatMapImage(:, :, jointNumber);
            [pos(1), pos(2)] = jointsAlgManager.findHeatMapMaximum(jointHeatMap);
        end
        
        function maxDistance = computeMaxJointDistance(jointsMatA, jointsMatB)
            squaredDistance = sum((jointsMatA - jointsMatB).^2, 2);
            maxDistance = max(sqrt(squaredDistance));
        end
        
        function [confidenceIndices, perJointAlgResults] = obtainAlgResultsForConfidenceImages(heatmapImages, isConfidentVector, jointNumber, startIndx)
            numOfImages = length(heatmapImages);
            confidenceIndices = find(isConfidentVector);
            confidentHeatMapImages = heatmapImages(confidenceIndices);
            perJointAlgResults = cell(numOfImages, 1);
            perJointAlgResults(confidenceIndices) = cellfun(@(x) jointsAlgManager.getJointLocationFromHeatMap(x, jointNumber), confidentHeatMapImages, 'UniformOutput', false);
            perJointAlgResults{startIndx} = jointsAlgManager.getJointLocationFromHeatMap(heatmapImages{startIndx}, jointNumber); %results for startIndx are taken from the heatMap from the heatmap even if it is not confident.
        end
        
    end
end
