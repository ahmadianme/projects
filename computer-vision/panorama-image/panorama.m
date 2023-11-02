clc;
clear;
close all;


% hyperparameters
harrisSigma = 2;
harrisThreshold = 0.05;
harrisRadius = 2;
neighborhoodRadius = 60;
matchesCount = 300;




currentDir = fileparts(mfilename('fullpath'));

imageL = im2double(imread(strcat(currentDir, '/images/image-left.jpg')));
imageR = im2double(imread(strcat(currentDir, '/images/image-right.jpg')));

grayImageL = rgb2gray(imageL);
grayImageR = rgb2gray(imageR);

[imageLHeight, imageLWidth, ~] = size(imageL);
[imageRHeight, imageRWidth, ~] = size(imageR);




[~, imageLRows, imageLColumns] = harris(grayImageL, harrisSigma, harrisThreshold, harrisRadius, 0);
[~, imageRRows, imageRColumns] = harris(grayImageR, harrisSigma, harrisThreshold, harrisRadius, 0);


figure;
imshow([imageL imageR]);
hold on;
plot(imageLColumns,imageLRows,'ys');
plot(imageRColumns + imageLWidth, imageRRows, 'ys'); 
title('Found feature points by harris.');
hold on;




imageLFeaturesDesc = findFeaturesDescription(grayImageL, neighborhoodRadius, imageLRows, imageLColumns);
imageRFeaturesDesc = findFeaturesDescription(grayImageR, neighborhoodRadius, imageRRows, imageRColumns);





[imageLMatchedIndicies, imageRMatchedIndicies] = matchFeatures(matchesCount, imageLFeaturesDesc, imageRFeaturesDesc);

imageLMatchedRows = imageLRows(imageLMatchedIndicies);
imageLMatchedColumns = imageLColumns(imageLMatchedIndicies);
imageRMatchedRows = imageRRows(imageRMatchedIndicies);
imageRMatchedColumns = imageRColumns(imageRMatchedIndicies);

figure;
imshow([imageL imageR]);
hold on;
plot(imageLMatchedColumns, imageLMatchedRows,'ys');
plot(imageRMatchedColumns + imageLWidth, imageRMatchedRows, 'ys'); 
title('Overlay top matched features');
hold on;




figR = [imageLMatchedRows, imageRMatchedRows];
figL = [imageLMatchedColumns, imageRMatchedColumns + imageLWidth];
figure; imshow([imageL imageR]);
hold on;
plot(imageLMatchedColumns, imageLMatchedRows,'ys');
plot(imageRMatchedColumns + imageLWidth, imageRMatchedRows, 'ys');
for i = 1:matchesCount 
    plot(figL(i,:), figR(i,:));
end
title('Top matched features mapping on images.');
hold on; 





[H, inlierIndices] = homographyEstimate([imageLMatchedColumns, imageLMatchedRows, ones(matchesCount,1)], [imageRMatchedColumns, imageRMatchedRows, ones(matchesCount,1)]);


imageLMatchedColumns = imageLMatchedColumns(inlierIndices);
imageRMatchedColumns = imageRMatchedColumns(inlierIndices);
imageLMatchedRows = imageLMatchedRows(inlierIndices);
imageRMatchedRows = imageRMatchedRows(inlierIndices);


figure;
imshow([imageL imageR]);
hold on;
plot(imageLMatchedColumns, imageLMatchedRows,'ys');
plot(imageRMatchedColumns + imageLWidth, imageRMatchedRows, 'ys'); 
title('Inlier Matches');
hold on;




img1Transformed = imtransform(imageL, maketform('projective', H));
figure;
imshow(img1Transformed);
title('Warped image');


stitchedCompositeImg = stitch(imageL, imageR, H);
figure;
imshow(stitchedCompositeImg);
title('Images aligned using homography');














function [cim, r, c] = harris(im, sigma, thresh, radius, disp)
    error(nargchk(2,5,nargin));
    
    dx = [-1 0 1; -1 0 1; -1 0 1]; % Derivative masks
    dy = dx';
    
    Ix = conv2(im, dx, 'same');    % Image derivatives
    Iy = conv2(im, dy, 'same');    

    % Generate Gaussian filter of size 6*sigma (+/- 3sigma) and of
    % minimum size 1x1.
    g = fspecial('gaussian',max(1,fix(6*sigma)), sigma);
    
    Ix2 = conv2(Ix.^2, g, 'same'); % Smoothed squared image derivatives
    Iy2 = conv2(Iy.^2, g, 'same');
    Ixy = conv2(Ix.*Iy, g, 'same');
    
    cim = (Ix2.*Iy2 - Ixy.^2)./(Ix2 + Iy2 + eps); % Harris corner measure

    % Alternate Harris corner measure used by some.  Suggested that
    % k=0.04 - I find this a bit arbitrary and unsatisfactory.
%   cim = (Ix2.*Iy2 - Ixy.^2) - k*(Ix2 + Iy2).^2; 

    if nargin > 2   % We should perform nonmaximal suppression and threshold
	
	% Extract local maxima by performing a grey scale morphological
	% dilation and then finding points in the corner strength image that
	% match the dilated image and are also greater than the threshold.
	sze = 2*radius+1;                   % Size of mask.
	mx = ordfilt2(cim,sze^2,ones(sze)); % Grey-scale dilate.
	cim = (cim==mx)&(cim>thresh);       % Find maxima.
	
	[r,c] = find(cim);                  % Find row,col coords.
	
	if nargin==5 & disp      % overlay corners on original image
	    figure, imagesc(im), axis image, colormap(gray), hold on
	    plot(c,r,'ys'), title('Detected corners by harris.');
	end
    
    else  % leave cim as a corner strength image and make r and c empty.
	r = []; c = [];
    end
end








function [ featDescriptions ] = findFeaturesDescription( img, radius, r, c )

    numFeat = length(r); %number of features
    featDescriptions = zeros(numFeat, (2 * radius + 1)^2);

    % matrix with a single 1 in the center and zeros all around it
    padHelper = zeros(2 * radius + 1); 
    padHelper(radius + 1, radius + 1) = 1;

    % use the pad Helper matrix to pad the img such that the border values
    % extend out by the radius
    paddedImg = imfilter(img, padHelper, 'replicate', 'full');

    %Extract the neighborhoods around the found features
    for i = 1 : numFeat
        % the indices held in r,c can be used as 
        %the top left corner of the neighborhood rather than its center
        rowRange = r(i) : r(i) + 2 * radius;
        colRange = c(i) : c(i) + 2 * radius;
        neighborhood = paddedImg(rowRange, colRange);
        flattenedFeatureVec = neighborhood(:);
        featDescriptions(i,:) = flattenedFeatureVec;
    end
    
    %Normalize all descriptors to have zero mean and unit standard deviation
    featDescriptions = zscore(featDescriptions')';
end









function [ H, inlierIndices ] = homographyEstimate( img1Feat, img2Feat )
%homographyEstimate Summary of this function goes here
%   Detailed explanation goes here

    parameters.numIterations = 150;      %the number of iterations to run
    parameters.subsetSize = 4;          %number of matches to use each iteration
    parameters.inlierDistThreshold = 10;   %the minimum distance for an inlier
    parameters.minInlierRatio = .3;     %minimum inlier ratio required to store a fitted model

    [H, inlierIndices] = ransac_H(parameters, img1Feat, img2Feat, @homographyFit, @calc_residuals);
    
    display('Number of inliers:');
    display(length(inlierIndices));
    display('Average residual for the inliers:')
    display(mean(calc_residuals(H, img1Feat(inlierIndices,:), img2Feat(inlierIndices,:))));
end











function H = homographyFit(pts1_homogenous, pts2_homogenous)

    if size(pts1_homogenous) ~= size(pts2_homogenous)
        error('Number of matched features in the subset supplied to homographyFit does not match for both images')
    end 
    
    [matchesCount, ~] = size(pts1_homogenous);
    
    %create the A matrix
    A = []; % will be 2*matchesCount x 9
    for i = 1:matchesCount
        %assume homogenous versions of all the feature points
        p1 = pts1_homogenous(i,:);
        p2 = pts2_homogenous(i,:);
        
        % 2x9 matrix to append onto A. 
        A_i = [ zeros(1,3)  ,   -p1     ,   p2(2)*p1;
                    p1      , zeros(1,3),   -p2(1)*p1];
        A = [A; A_i];        
    end
    
    %solve for A*h = 0
    [~,~,eigenVecs] = svd(A); % Eigenvectors of transpose(A)*A
    h = eigenVecs(:,9);     % Vector corresponding to smallest eigenvalue 
    H = reshape(h, 3, 3);   % Reshape into 3x3 matrix
    H = H ./ H(3,3);        % Divide through by H(3,3)
    
end










function residuals = calc_residuals(H, homoCoord1, homoCoord2)
%CALC_RESIDUALS Summary of this function goes here
%   Detailed explanation goes here

    %transform the points from img 1 by multiplying the homo coord by H
    transformedPoints = homoCoord1 * H;
    
    %divide each pt by 3rd coord (scale factor lambda) to yield [x;y;1]
    %before taking difference
    lambda_t =  transformedPoints(:,3); %scale factor
    lambda_2 = homoCoord2(:,3);    %scale factor 
    cartDistX = transformedPoints(:,1) ./ lambda_t - homoCoord2(:,1) ./ lambda_2;
    cartDistY = transformedPoints(:,2) ./ lambda_t - homoCoord2(:,2) ./ lambda_2;
    residuals = cartDistX .* cartDistX + cartDistY .* cartDistY;
end










function [cartCoord] = homo_2_cart(homoCoord)
%UNHOMOGENIZE_COORDINATES Summary of this function goes here
%   Detailed explanation goes here

    dimension = size(homoCoord, 2) - 1;
        
    %divide every row by the last entry in that row
    normCoord = bsxfun(@rdivide,homoCoord,homoCoord(:,end));
    cartCoord = normCoord(:,1:dimension);
end







function [ img1Feature_idx, img2Feature_idx ] = matchFeatures( matchesCount, featDescriptions1, featDescriptions2)
    %determine the dist between every pair of features between images
    %(ie: every combination of 1 feature from img1 and 1 feature from img2)
    distances = dist2(featDescriptions1, featDescriptions2);
    %sort these distances
    [~,distance_idx] = sort(distances(:), 'ascend');
    %select the smallest distances as the best matches
    bestMatches = distance_idx(1:matchesCount);
    % Determine the row,col indices in the distances matrix containing the best
    % matches, as they'll be used to determine which feature pair produced that 
    % distance. The distances matrix is m x n where m = numFeaturesImg1 and 
    % n = numFeaturesImg2... so we access img1 feature as the row and img2
    % feature as the col
    [rowIdx_inDistMatrix, colIdx_inDistMatrix] = ind2sub(size(distances), bestMatches);
    img1Feature_idx = rowIdx_inDistMatrix;
    img2Feature_idx = colIdx_inDistMatrix;
end











function n2 = dist2(x, c)
% DIST2	Calculates squared distance between two sets of points.
% Adapted from Netlab neural network software:
% http://www.ncrg.aston.ac.uk/netlab/index.php
%
%	Description
%	D = DIST2(X, C) takes two matrices of vectors and calculates the
%	squared Euclidean distance between them.  Both matrices must be of
%	the same column dimension.  If X has M rows and N columns, and C has
%	L rows and N columns, then the result has M rows and L columns.  The
%	I, Jth entry is the  squared distance from the Ith row of X to the
%	Jth row of C.
%	Copyright (c) Ian T Nabney (1996-2001)

    [ndata, dimx] = size(x);
    [ncentres, dimc] = size(c);
    if dimx ~= dimc
        error('Data dimension does not match dimension of centres')
    end

    n2 = (ones(ncentres, 1) * sum((x.^2)', 1))' + ...
      ones(ndata, 1) * sum((c.^2)',1) - ...
      2.*(x*(c'));

    % Rounding errors occasionally cause negative entries in n2
    if any(any(n2<0))
      n2(n2<0) = 0;
    end
end









function [ bestFitModel, inlierIndices ] = ransac_H( parameters, x, y, fitModelFxn, errorFxn )

    [matchesCount, ~] = size(x);
    numInliersEachIteration = zeros(parameters.numIterations,1);
    storedModels = {};%zeros(parameters.numIterations,3,3);
    
    for i = 1 : parameters.numIterations
        %display(['Running ransac Iteration: ', num2str(i)]);
        
        %select a random subset of points
        subsetIndices = randsample(matchesCount, parameters.subsetSize);
        x_subset = x(subsetIndices, :);
        y_subset = y(subsetIndices, :);
            
        %fit a model to that subset
        model = fitModelFxn(x_subset, y_subset);
        
        %compute inliers, ie: find all remaining points that are 
        %"close" to the model and reject the rest as outliers
        residualErrors = errorFxn(model, x, y);
        
        %display(['Mean Residual Error: ', num2str(mean(residualErrors))]);
        inlierIndices = find(residualErrors < parameters.inlierDistThreshold);      

        %record the number of inliers
        numInliersEachIteration(i) = length(inlierIndices);
        
        %keep track of any models that generated an acceptable numbers of 
        %inliers. This collection can be parsed later to find the best fit
        currentInlierRatio = numInliersEachIteration(i)/matchesCount;
        if currentInlierRatio >=  parameters.minInlierRatio
        %if numInliersEachIteration(i) >= max(numInliersEachIteration)
            %re-fit the model using all of the inliers and store it
            x_inliers = x(inlierIndices, :);
            y_inliers = y(inlierIndices, :);
            storedModels{i} = fitModelFxn(x_inliers, y_inliers);
        end
    end
    %display(storedModels);
    %display(numInliersEachIteration);
    
    %retrieve the model with the best fit (highest number of inliers)
    bestIteration = find(numInliersEachIteration == max(numInliersEachIteration));
    bestIteration = bestIteration(1); %incase there was more than 1 with same value
    bestFitModel = storedModels{bestIteration};
    
    %recalculate the inlier indices for all points, this was done once before 
    %when calculting this model, but it wasn't stored for space reasons. 
    %Recalculate it now so that it can be returned to the caller
    residualErrors = errorFxn(bestFitModel, x, y);
    inlierIndices = find(residualErrors < parameters.inlierDistThreshold);
end







function [composite] = stitch(im1, im2, H)

    [h1, w1, numChannels1] = size(im1);
    [h2, w2, numChannels2] = size(im2);
    %create a matrix of corner points for the first image
    corners = [ 1 1 1;
                w1 1 1;
                w1 h1 1;
                1 h1 1];
    %warp the corner points using the homography matrix    
    warpCorners = homo_2_cart( corners * H );

    %determine the minimum and maximum bounds for the composite image based off
    %the warped corners
    minX = min( min(warpCorners(:,1)), 1);
    maxX = max( max(warpCorners(:,1)), w2);
    minY = min( min(warpCorners(:,2)), 1);
    maxY = max( max(warpCorners(:,2)), h2);

    %use those min and max bounds to define the resolution of the composite image
    xResRange = minX : maxX; %the range for x pixels
    yResRange = minY : maxY; %the range for y pixels

    [x,y] = meshgrid(xResRange,yResRange) ;
    Hinv = inv(H);

    warpedHomoScaleFactor = Hinv(1,3) * x + Hinv(2,3) * y + Hinv(3,3);
    warpX = (Hinv(1,1) * x + Hinv(2,1) * y + Hinv(3,1)) ./ warpedHomoScaleFactor ;
    warpY = (Hinv(1,2) * x + Hinv(2,2) * y + Hinv(3,2)) ./ warpedHomoScaleFactor ;


    if numChannels1 == 1
        %images are black and white... so simple interpolation
        blendedLeftHalf = interp2( im2double(im1), warpX, warpY, 'cubic') ;
        blendedRightHalf = interp2( im2double(im2), x, y, 'cubic') ;
    else
        %images are RGB, so interpolate each channel individually
        blendedLeftHalf = zeros(length(yResRange), length(xResRange), 3);
        blendedRightHalf = zeros(length(yResRange), length(xResRange), 3);
        for i = 1:3
            blendedLeftHalf(:,:,i) = interp2( im2double( im1(:,:,i)), warpX, warpY, 'cubic');
            blendedRightHalf(:,:,i) = interp2( im2double( im2(:,:,i)), x, y, 'cubic');
        end
    end
    %create a blend weight matrix based off the presence of a pixel value from
    %either image in the composite... ie: overlapping region has blendweight of
    %2, a non overlapping region of 1 img has a blendweight of 1, and a region
    %with no img (blank space) has a blendweight of 0.
    blendWeight = ~isnan(blendedLeftHalf) + ~isnan(blendedRightHalf) ;
    %replace all NaN with 0, so they can be blended properly even if there is
    %no pixel value there
    blendedLeftHalf(isnan(blendedLeftHalf)) = 0 ;
    blendedRightHalf(isnan(blendedRightHalf)) = 0 ;
    %add the blendedLeft and Right halves together while dividing by the
    %blendWeight for that pixel.
    composite = (blendedLeftHalf + blendedRightHalf) ./ blendWeight ;

end



