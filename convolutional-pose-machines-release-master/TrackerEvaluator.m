classdef TrackerEvaluator < handle
    %% Public Methods
    methods (Static)
        function resultsPerJoint = runPerJointMDnetOnVideo(kinectDateVector)
            [jointMatFileName, resultFileName, videoPath] = TrackerEvaluator.createFileNamesFromDateVector(kinectDateVector, false);
            load(jointMatFileName);
            [img_files, filenums] = TrackerEvaluator.retrieveFileInformation(videoPath);
            TrackerEvaluator.addAllMDNetPaths();
            img_files = fullfile(videoPath, img_files);
            posWidth = 50;
            posHeight = 50;
            trackingDisplay = true;
            initIndex = 3;%not 1 to get more stable joint estimation from kinect
            pos = jointLists{filenums(initIndex),1}; %#ok<USENS>
            pos  = kinectHandler.blobJointsInKinectCoords(pos, [256 256], [480 640]);
            % pos = [pos(:, 2) pos(:, 1)];
            resultsPerJoint = cell(7,1);
            net = fullfile('/home/felix/BGU_Computer_Vision_thesis/Codes/Pose-Estimation-Using-Tracking/MDNet-master','models','mdnet_vot-otb.mat');
            for k = 1:length(resultsPerJoint)
                initPos = pos(k,:);
                posRect = [initPos(1)- posWidth/2 initPos(2)-posHeight/2 posWidth posHeight];
                resultsPerJoint{k} = mdnet_run(img_files(initIndex:end), posRect, net, trackingDisplay);
            end
            save(resultFileName, 'resultsPerJoint');
        end
        
        function resultsPerJoint = runPerJointKCFOnVideo(kinectDateVector)
            [jointMatFileName, resultFileName, videoPath] = TrackerEvaluator.createFileNamesFromDateVector(kinectDateVector, true);
            addpath(genpath('kcf_tracker'));
            load(jointMatFileName);
            [img_files, filenums] = TrackerEvaluator.retrieveFileInformation(videoPath);
            kernel.type ='gaussian';
            output_sigma_factor = 0.1;
            features.hog_orientations = 9;
            
            %Parameters for gray-scale tracker.
            interp_factor = 0.075;  %linear interpolation factor for adaptation
            
            kernel.sigma = 0.2;  %gaussian kernel bandwidth
            
            kernel.poly_a = 1;  %polynomial kernel additive term
            kernel.poly_b = 7;  %polynomial kernel exponent
            
            features.gray = true;
            features.hog = false;
            cell_size = 1;
            padding = 1.5;
            lambdaq = 10^(-4);
            posWidth = 50;
            posHeight = 50;
            initIndex = 3;%not 1 to get more stable joint estimation from kinect
            pos = jointLists{filenums(initIndex),1}; %#ok<USENS>
            pos  = kinectHandler.blobJointsInKinectCoords(pos, [256 256], [480 640]);
            % pos = [pos(:, 2) pos(:, 1)];
            resultsPerJoint = cell(7,1);
            for k = 1:length(resultsPerJoint)
                initPos = pos(k,:);
                [resultsPerJoint{k}, ~] = tracker(videoPath, img_files(initIndex:end), initPos, [posWidth posHeight], ...
                    padding, kernel, lambdaq, output_sigma_factor, interp_factor, cell_size, features, false);
            end
            save(resultFileName, 'resultsPerJoint');
        end
        
        function [img_files, filenums] = retrieveFileInformation(videoPath)
            videoFolderInfo = dir([videoPath '*.jpg']);
            img_files = {videoFolderInfo.name}';
            numOfImageFiles = length(img_files);
            filenums = 1:numOfImageFiles;
            for k = 1:numOfImageFiles
                C = strsplit(img_files{k}, '.');
                filenums(k) = str2double(C{1});
            end
            filenums = sort(filenums);
            img_files = num2str(filenums','%d');
            img_files = strcat(num2cell(img_files,2), '.jpg');
            img_files = strtrim(img_files);
        end
        
        
        function [jointMatFileName, resultFileName, videoPath, cnnResultFileName, otherTrackerResultsFile] = createFileNamesFromDateVector(kinectDateVector, isKCFTracker)
            kinectDateCell = num2cell(kinectDateVector, 1);
            kinectDateStr = sprintf('%02d_%02d_%02d_%02d_%02d', kinectDateCell{:});
            jointMatFileName = sprintf('kinectJointsData%s.mat', kinectDateStr);
            if (isKCFTracker)
                resultFileName = sprintf('resultsKCF%s.mat', kinectDateStr);
                otherTrackerResultsFile = sprintf('results%s.mat', kinectDateStr);
            else
                resultFileName = sprintf('results%s.mat', kinectDateStr);
                otherTrackerResultsFile = sprintf('resultsKCF%s.mat', kinectDateStr);
            end
            cnnResultFileName = sprintf('CNNresults%s.mat', kinectDateStr);
            videoPath = sprintf('kinectVideoFrames_%s/', kinectDateStr);
        end
        
        
        function trackerMetric = computeTrackerMetric4AllJoints(videoPath, trackerMatFileName, jointMatFileName, isMDnetTracker, startIndex)
            load(jointMatFileName);
            load(trackerMatFileName, 'resultsPerJoint');
            
            if (isMDnetTracker)
                resultsPerJoint = TrackerEvaluator.convertMDNetResult2JointResultsFormat(resultsPerJoint);
            end
            
            jointListsKinect = kinectHandler.getOnlyRelevantKinectJoints(videoPath, startIndex, jointLists);
            numOfFrames = size(resultsPerJoint{1}, 1);
            trackerMetric = zeros(numOfFrames, 1);
            
            for k = 1:numOfFrames
                %compute jointMats
                jointMatTracker = TrackerEvaluator.computeJointMatFromTrackerResults(resultsPerJoint, k);
                jointListsKinect{k} = kinectHandler.blobJointsInKinectCoords(jointListsKinect{k}, [256 256], [480 640]); %convert back from Zisserman's CNN coords.
                %compute metrics:
                trackerMetric(k) = jointsAlgManager.computeMaxJointDistance(jointMatTracker, jointListsKinect{k});
            end
        end
        
        
        %% Continuing Live Code.
        function trackingResults =  performSimpleMDNetTracking(videoPath, initPos, posWidth, posHeight, net, img_files, trackingDisplay)
            img_files = fullfile(videoPath, img_files);
            TrackerEvaluator.addAllMDNetPaths();
            posRect = [initPos(1)- posWidth/2 initPos(2)-posHeight/2 posWidth posHeight];
            trackingResults = mdnet_run(img_files, posRect, net, trackingDisplay);
            trackingResults = trackingResults(:, 1:2);
        end
        
        function kcfResults = performSimpleKCFTracking(videoPath ,initPos, posWidth, posHeight, img_files, trackingDisplay)
            kernel.type ='gaussian';
            output_sigma_factor = 0.1;
            features.hog_orientations = 9;
            
            %Parameters for gray-scale tracker.
            interp_factor = 0.075;  %linear interpolation factor for adaptation
            
            kernel.sigma = 0.2;  %gaussiakinectDateVectorn kernel bandwidth
            
            kernel.poly_a = 1;  %polynomial kernel additive term
            kernel.poly_b = 7;  %polynomial kernel exponent
            
            features.gray = true;
            features.hog = false;
            cell_size = 1;
            padding = 1.5;
            lambdaq = 10^(-4);
            [kcfResults, ~] = tracker(videoPath, img_files, initPos, [posWidth posHeight], ...
                padding, kernel, lambdaq, output_sigma_factor, interp_factor, cell_size, features, trackingDisplay);
            
        end
        
        
    end
    %% Private Methods
    methods(Static, Access = private)
        
        function jointsTrackerFormat = transferCNNJoints2TrackerFormat(joints)
            jointsTrackerFormat = cell(7,1);
            for k = 1:length(jointsTrackerFormat)
                coordsPerJoint =  double(joints(:, k, :));
                coordsPerJoint = reshape(coordsPerJoint, 2, size(joints, 3));
                coordsPerJoint = coordsPerJoint';
                coordsPerJoint = kinectHandler.blobJointsInKinectCoords(coordsPerJoint, [256 256], [480 640]);
                jointsTrackerFormat{k} = coordsPerJoint;
            end
        end
        
        
        
        function jointMat = computeJointMatFromTrackerResults(jointResultsTrackerFormat, frameNum)
            numOfJoints = length(jointResultsTrackerFormat);
            jointMat = zeros(numOfJoints,2);
            for m = 1:numOfJoints
                perJointMatKCF =  jointResultsTrackerFormat{m};
                jointMat(m,:) = perJointMatKCF(frameNum,:);
            end
        end
        
        function resultsPerJointMDNet = convertMDNetResult2JointResultsFormat(resultsPerJoint)
            resultsPerJointMDNet = cell(size(resultsPerJoint));
            for k = 1:length(resultsPerJointMDNet)
                perJointMat = resultsPerJoint{k};
                perJointMat = [perJointMat(:,1) + perJointMat(:,3)/2  perJointMat(:,2) + perJointMat(:,4)/2];
                resultaddAllMDNetPathssPerJointMDNet{k} = perJointMat;
            end
        end
        
        function addAllMDNetPaths()
            addpath('/home/felix/BGU_Computer_Vision_thesis/Codes/Pose-Estimation-Using-Tracking/MDNet-master/pretraining');
            addpath('/home/felix/BGU_Computer_Vision_thesis/Codes/Pose-Estimation-Using-Tracking/MDNet-master/tracking');
            addpath('/home/felix/BGU_Computer_Vision_thesis/Codes/Pose-Estimation-Using-Tracking/MDNet-master/utils');
        end
        
        
    end
    
    
    
end