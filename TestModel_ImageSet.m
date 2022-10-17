clc
clear
close all

%% Load test images and feature extraction.

% Create Image data store.
imds = imageDatastore('TestData', ...
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames');

% Total number of images.
numImages = numel(imds.Labels);

% Name of each class.
classNames = categories(imds.Labels);

for i = 1:numImages
    
    I = readimage(imds,i);
    if size(I,3) == 3
        I = rgb2gray(I);
    end
    
    Labels(i) = double(imds.Labels(i))-1;
    
    % Detect eyes and mouth in each image.
    eyesDetector = vision.CascadeObjectDetector('EyePairBig');
    mouthDetector = vision.CascadeObjectDetector('Mouth', 'MergeThreshold', 15);
    noseDetector = vision.CascadeObjectDetector('Nose', 'MergeThreshold', 10);
    
    %Find HOG and LBP features in each image.
    % eyes
    eyeFind = step(eyesDetector, I);
    foundEyesIm = insertObjectAnnotation(I, 'rectangle', eyeFind, 'Eyes');
    if size(eyeFind,1)~= 1
        x = eyeFind(1,:);
        clear eyeFind
        eyeFind = x;
        clear x
    end
    eyeBox = imcrop(I,eyeFind);
    eyeBox = imresize(eyeBox, [11 45]);%[20 70]
    HOGeyes = extractHOGFeatures(eyeBox,'CellSize',[4 4]);
    
    % mouth
    mouthFind = step(mouthDetector, I);
    foundMouthIm = insertObjectAnnotation(I, 'rectangle', mouthFind, 'Mouth');
    if size(mouthFind,1)~= 1
        x = mouthFind(1,:);
        clear mouthFind
        mouthFind = x;
        clear x
    end
    mouthBox = imcrop(I,mouthFind);
    mouthBox = imresize(mouthBox, [15 25]);%[50 60]
    HOGmouth = extractHOGFeatures(mouthBox,'CellSize',[4 4]);
    
    % nose
    noseFind = step(noseDetector, I);
    foundNoseIm = insertObjectAnnotation(I, 'rectangle', noseFind, 'Nose');
    if size(noseFind,1)~= 1
        x = noseFind(1,:);
        clear noseFind
        noseFind = x;
        clear x
    end
    noseBox = imcrop(I,noseFind);
    noseBox = imresize(noseBox, [15 18]);
    HOGnose = extractHOGFeatures(noseBox,'CellSize',[4 4]);
    
    % Creat featere matrix.
    Features(i,:) = [HOGeyes, HOGmouth, HOGnose];
    
    fprintf('Processed %0.1f percent records\n',(i/numImages)*100);
    
end
    
%% Feature reduction.
active_CHI = 1;
if active_CHI
    load('idx.mat');
    Features = Features(:,idx(1:110));
end

%% Test trained model.
load('TrainedModel.mat');
[labelSVM,scoreSVM] = predict(SVMModel,Features);
accuracy = (sum(labelSVM' == Labels)/(length(Labels)))*100;
figure
confusionchart(Labels',labelSVM,'RowSummary','row-normalized','ColumnSummary','column-normalized')
