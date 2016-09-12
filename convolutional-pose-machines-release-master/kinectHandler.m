classdef kinectHandler < handle
    %% Public
    methods (Static)
        
        function [joint4Validation, rgbFrame] = collectDataUsingKinect(vidRGB, vidDepth, timeLap, totNumOfFrames)
            disp('Acquisition Starts in 3 seconds.');
            pause(3);
            start([vidRGB vidDepth]);
            pause(1);
            trigger([vidRGB vidDepth]);
            while get(vidDepth,'FramesAvailable') < totNumOfFrames  %Wait until at least 1 frame is available
            end
            [~, ~, metaData] = getdata(vidDepth);
%              while get(vidDepth,'FramesAvailable')<1  %Wait until at least 1 frame is available
%             end
            [rgbFrame, ~, ~] = getdata(vidRGB);
            disp('Acquisition Ended');
            jointLists = kinectHandler.perviewSkeletonData(metaData, rgbFrame, timeLap);
            stop(vidDepth);
            stop(vidRGB);
            joint4Validation  = cellfun(@(x) kinectHandler.getJointsInNNFormat(x, size(rgbFrame), [256 256]), jointLists,'UniformOutput', false);
        end
        
        function [jointLists, validJointsLists] = kinectMain(totNumOfFrames, timeLap)
            imaqreset;
            clc;
            disp('Press a key to start Kinect acquisition');
            pause;
            [vidDepth, vidRGB] = kinectHandler.InitializeKinect(totNumOfFrames);
            [jointLists, collectedFrames] = kinectHandler.collectDataUsingKinect(vidRGB, vidDepth, timeLap, totNumOfFrames);
            validJointsLists = ~cellfun(@isempty, jointLists);
            dateString4File = datestr(now, 'dd_mm_yy_HH_MM');
            imageFolderName = ['kinectVideoFrames_' dateString4File];
            mkdir(imageFolderName);
            kinectHandler.writeKinectImages2Folder(validJointsLists, collectedFrames, imageFolderName);
            save(['kinectJointsData' dateString4File '.mat'],'jointLists', 'validJointsLists');
        end
        
        function distancesPerJoints = computeDistanceInPixels(jointsNN, jointsKinectNNFormat)
            squaredDiffs = (jointsNN - jointsKinectNNFormat).^2;
            squaredDistances = sum(squaredDiffs, 1);
            distancesPerJoints = sqrt(squaredDistances);
        end
        
        function plotKinectSkeletons(skeletons, imageFolderName, frameRange, blobSize)
            for k = frameRange
                fileName = sprintf('%d.jpg',k);
                imageFilePath = fullfile(imageFolderName, fileName);
                I =imread(imageFilePath);
                imResized = imresize(I, [blobSize blobSize]);
                figure(1)
                imshow(imResized);
                hold on
                plotSkeleton(skeletons{k}', [], []);
                hold off
                pause(0.05);
            end
        end
        
         function plotCNNSkeletons(skeletons, imageFolderName, imageFiles, blobSize)
            for k = 1 : length(imageFiles)
                imageFilePath = fullfile(imageFolderName, imageFiles{k});
                I =imread(imageFilePath);
                imResized = imresize(I, [blobSize blobSize]);
                figure(1)
                imshow(imResized);
                hold on
                plotSkeleton(skeletons(:,:,k), [], []);
                hold off
                pause(0.05);
            end
        end
        
        function writeKinectImages2Folder(validityVec, imageMatrix, outputFolder)
            for k = 1:size(imageMatrix,4)
                if (validityVec(k))
                    fileName = sprintf('%d.jpg', k);
                    imwrite(imageMatrix(:,:,:,k), fullfile(outputFolder ,fileName));
                end
            end
        end
        
        function [vidDepth, vidRGB] = InitializeKinect(totNumOfFrames)
            vidRGB = videoinput('kinect',1,'RGB_640x480');
            vidDepth = videoinput('kinect',2,'Depth_640x480');
            src2 = getselectedsource(vidDepth);
            set(src2,'BodyPosture','Seated');
            set(src2,'TrackingMode','Skeleton');
            set(vidDepth,'FramesPerTrigger', totNumOfFrames);
            set(vidRGB,'FramesPerTrigger', totNumOfFrames);
            triggerconfig(vidRGB, 'manual');
            set(vidRGB,'TriggerRepeat',inf);
            triggerconfig(vidDepth, 'manual');
            set(vidDepth,'TriggerRepeat',inf);
        end
        
        function jointsNNFormat = getJointsInNNFormat(jointCoords, imSize, blobSize)
            jointsNNFormat = jointCoords;
            if (~isempty(jointCoords))
            jointBlobCoords  = kinectHandler.kinectJointsInBlobCoords(jointCoords, imSize, blobSize);
            jointsNNFormat = kinectHandler.convert10jointsTo7(jointBlobCoords);
            end
        end
        
         function jointBlobCoords  = kinectJointsInBlobCoords(jointKinectCoords, imSize, blobSize)
            imSizeX = imSize(2);
            imSizeY = imSize(1);
            blobSizeX = blobSize(2);
            blobSizeY = blobSize(1);
            jointBlobCoords(:,1) = jointKinectCoords(:,1) * blobSizeX/imSizeX;
            jointBlobCoords(:,2) = jointKinectCoords(:,2) * blobSizeY/imSizeY;
        end
        
        function jointBlobCoords  = blobJointsInKinectCoords(jointBlobCoords, blobSize, imSize)
            imSizeX = imSize(2);
            imSizeY = imSize(1);
            blobSizeX = blobSize(2);
            blobSizeY = blobSize(1);
            jointBlobCoords(:,1) = jointBlobCoords(:,1) * imSizeX/blobSizeX;
            jointBlobCoords(:,2) = jointBlobCoords(:,2) * imSizeY/blobSizeY;
        end
        
        function jointListsKinect = getOnlyRelevantKinectJoints(videoPath, startIndex, jointListsKinect)
            [~, filenums] = TrackerEvaluator.retrieveFileInformation(videoPath);
            jointListsKinect = jointListsKinect(filenums(startIndex):end, 1);
        end 
        
    end
    %% Private
    methods (Static, Access = private)
          
        function jointLists =  perviewSkeletonData(kinectMetaData, Frame, timeLap)
            jointLists = cell(length(kinectMetaData), 1);
            for k=1:length(kinectMetaData)
                jointLists{k} = kinectHandler.skeletonViewer(kinectMetaData(k).JointImageIndices, Frame(:,:,:,k), kinectMetaData(k).IsSkeletonTracked);
                pause(timeLap);
            end
        end
        
        function jointMat7 = convert10jointsTo7 (jointMat10)
            jointReorder = [2, 5, 9, 4, 8, 3, 7]; %joint order in CNN(1 is joint No. 3 in Kinect, 2 is joint No.4 and so on).
            jointMat7 = jointMat10(jointReorder, :);
        end
        
        
        function returnedSkeleton = skeletonViewer(skeleton, image, IsSkeletonTracked)
            skeleton = skeleton(3:12,:,:);
            imshow(image,[]);
            SkeletonConnectionMap = [[3 4];
                [3 5]; %Left Hand
                [5 6];
                [6 7];
                [7 8];
                [3 9]; %Right Hand
                [9 10];
                [10 11];
                [11 12]];
            SkeletonConnectionMap = SkeletonConnectionMap - 2;
            skeletonIndx = find(IsSkeletonTracked,1);
            for i = 1: size(SkeletonConnectionMap, 1)
                if nnz(IsSkeletonTracked) > 0
                    X1 = [skeleton(SkeletonConnectionMap(i,1), 1, skeletonIndx) skeleton(SkeletonConnectionMap(i,2), 1, skeletonIndx)];
                    Y1 = [skeleton(SkeletonConnectionMap(i,1), 2, skeletonIndx) skeleton(SkeletonConnectionMap(i,2), 2, skeletonIndx)];
                    line(X1,Y1, 'LineWidth', 1.5, 'LineStyle', '-', 'Marker', '+', 'Color', 'r');
                    returnedSkeleton = skeleton(:, :, skeletonIndx);
                else
                    returnedSkeleton = [];
                end
                hold on;
            end
            hold off;
            
        end
      
        
    end
end