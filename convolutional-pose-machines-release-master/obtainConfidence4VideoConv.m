function [isConfidentVector, peakProbabilities, probabilitiesInRadius] = obtainConfidence4VideoConv(heatmapCellArray, confidenceAlgParams, jointNumber)
      
      relativeRadius = confidenceAlgParams.relativeRadius;
      peakProbabiltyThresh = confidenceAlgParams.peakProbabiltyThresh;
      probabiltyInRadiusThresh = confidenceAlgParams.probabiltyInRadiusThresh;
      avergingWindowSize = confidenceAlgParams.avergingWindowSize;
      confidenceThresh = confidenceAlgParams.confidenceThresh;
      
      numOfFrames = length(heatmapCellArray);
      avgWindow = ones(avergingWindowSize, 1);
      [isConfidentVector, peakProbabilities, probabilitiesInRadius] = deal(zeros(numOfFrames, 1));
      for k = 3:numOfFrames
           heatmapImages = heatmapCellArray{k};
           heatmapImage = heatmapImages(: ,: ,jointNumber);
                [~ , peakProbabilities(k), probabilitiesInRadius(k)] = jointsAlgManager.checkIfConfidentJoint(heatmapImage, relativeRadius, peakProbabiltyThresh, probabiltyInRadiusThresh);
      end
      
      peakProbabilitiesAverages = conv(avgWindow, peakProbabilities,'same');
      probabilitiesInRadiusAverages =  conv(avgWindow, probabilitiesInRadius,'same');
      isConfidentVector(1: avergingWindowSize) = 1;
      logIndicesOfConfidentFrames = peakProbabilities > confidenceThresh * peakProbabilitiesAverages & probabilitiesInRadius > confidenceThresh * probabilitiesInRadiusAverages;
      isConfidentVector(avergingWindowSize + 1:end) =  logIndicesOfConfidentFrames(avergingWindowSize + 1:end); %first part is thresh because of the convolution.
      
end
