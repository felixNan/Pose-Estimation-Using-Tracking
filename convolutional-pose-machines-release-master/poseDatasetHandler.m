classdef poseDatasetHandler < handle
    %% Public
    methods (Static)
        
        function [sequencesOfImages, sequencesOfImageNames] = CollectVideoSequencesFomMPII(mpiiImageFolder, imageFormat)
            default imageFormat = {'*.jpg'}
            imageFileiInfo = dir([mpiiImageFolder filesep imageFormat]);
            imageFileNames = {imageFileiInfo.name}';
            numOfImageFiles = length(imageFileNames);
			imageNumbers = zeros(numOfImageFiles, 1);
			imageNames = cell(numOfImageFiles, 1);
            for k = 1:numOfImageFiles
				[~, imageNames{k},~] = fileparts(imageFileNames{k});
				imageNameParts = strsplit(imageNames{k}, '_');
				imageNumbers(k)= str2double(imageNameParts{2});
            end
            [sortedImageNums, sortedIndices] = sort(imageNumbers, 'ascend');
            sortedImageFileNames = imageFileNames(sortedIndices);
            sortedImageNames = imageNames(sortedIndices);
            videoEdges = find(diff([0;sortedImageNums]) > 1);
            rangesOfVideoSequences = [videoEdges(1:end -1) ; videoEdges(2:end)];
            numOfSequences = size(rangesOfVideoSequences, 1);
            [sequencesOfImages, sequencesOfImageNames]  = cell(numOfSequences,1);
            for m = 1:numOfSequences
				frameRange = rangesOfVideoSequences(m,1): rangesOfVideoSequences(m,2);  
				sequencesOfImages{m} = sortedImageFileNames(frameRange);
				sequencesOfImageNames{m} = sortedImageNames(frameRange);
            end
        end
        
        function jointListsDataSequence = GetMPIIJointListDataForSequencesOfImages(sequencesOfImageNames, gtDataFolder)
			numOfSequences = length(sequencesOfImageNames);
			jointListsDataSequence = cell(numOfSequences, 1);
			for m = 1: numOfSequences
				listOfImageNames = sequencesOfImageNames{m};
				dataFilePaths = fullfile(gtDataFolder, listOfImageNames, '.mat' );
				jointListsDataSequence{m} =  poseDatasetHandler.convertDataFileListToJointList4MPII(dataFilePaths);
			end
        end
        
        
    end
    %% Private
    methods (Static, Access = private)
          
        function jointListData =  convertDataFileListToJointList4MPII(dataFilePathList)
			jointListData = cellfun(@load, dataFilePathList,'uniformOutput' ,false);
			jointListData = cellfun(@poseDatasetHandler.convertMPIIjointsTo7, jointListData, 'uniformOutput' ,false);
        end
        
        function jointMat7 = convertMPIIjointsTo7 (jointMat10)
            jointReorder = [2, 5, 9, 4, 8, 3, 7]; %joint order in CNN(1 is joint No. 3 in Kinect, 2 is joint No.4 and so on).
            jointMat7 = jointMat10(jointReorder, :);
        end
     
        
    end
end
