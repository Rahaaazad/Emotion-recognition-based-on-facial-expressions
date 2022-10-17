 clc
clear
close all

%% Take test image.
Cam = webcam;
preview(Cam)
pause()
ImgTest = snapshot(Cam);
closePreview(Cam)
figure(1); imshow(ImgTest)

%% Preprocess test image.
ImgT = rgb2gray(ImgTest);

faceDetector = vision.CascadeObjectDetector('FrontalFaceCART', 'MergeThreshold', 5);
faceBox = step(faceDetector, ImgT);
faceFind = insertObjectAnnotation(ImgT, 'rectangle', faceBox, 'Face');
face = imcrop(ImgT,faceBox);

face = imresize(face, [100 100]);
figure(); imshow(face)

%% Extract features of test image.

% Detect eyes and mouth in each image.
    eyesDetector = vision.CascadeObjectDetector('EyePairBig');
    mouthDetector = vision.CascadeObjectDetector('Mouth', 'MergeThreshold', 15);
    noseDetector = vision.CascadeObjectDetector('Nose', 'MergeThreshold', 10);
    
    %Find HOG and LBP features in each image.
    % eyes
    eyeFind = step(eyesDetector, face);
    foundEyesIm = insertObjectAnnotation(face, 'rectangle', eyeFind, 'Eyes');
    if size(eyeFind,1)~= 1
        x= eyeFind(1,:);
        clear eyeFind
        eyeFind = x;
        clear x
    end
    eyeBox = imcrop(face,eyeFind);
    eyeBox = imresize(eyeBox, [11 45]);
    HOGeyes = extractHOGFeatures(eyeBox,'CellSize',[4 4]);
%     LBPeyes = extractLBPFeatures(eyeBox);
    
    % mouth
    mouthFind = step(mouthDetector, face);
    foundMouthIm = insertObjectAnnotation(face, 'rectangle', mouthFind, 'Mouth');
    if size(mouthFind,1)~= 1
        x = mouthFind(1,:);
        clear mouthFind
        mouthFind = x;
        clear x
    end
    mouthBox = imcrop(face,mouthFind);
    mouthBox = imresize(mouthBox, [15 25]);
    HOGmouth = extractHOGFeatures(mouthBox,'CellSize',[4 4]);
%     LBPmouth = extractLBPFeatures(mouthBox);


    % nose
    noseFind = step(noseDetector, face);
    foundNoseIm = insertObjectAnnotation(face, 'rectangle', noseFind, 'Nose');
    if size(noseFind,1)~= 1
        x = noseFind(1,:);
        clear noseFind
        noseFind = x;
        clear x
    end
    noseBox = imcrop(face,noseFind);
    noseBox = imresize(noseBox, [15 18]);
    HOGnose = extractHOGFeatures(noseBox,'CellSize',[4 4]);
    
    % Creat featere matrix.
    Features = [HOGeyes, HOGmouth, HOGnose]; 
    
%% Feature reduction.
active_CHI = 1;
if active_CHI
    load('idx.mat');
    Features = Features(:,idx(1:110));
end

%% Test trained model.
load('TrainedModel.mat');
[labelSVM,scoreSVM] = predict(SVMModel,Features);
accuracy = (1+max(scoreSVM))*100;
switch labelSVM
    case 0
        text_str = 'Anger ';
    case 1
        text_str = 'Disgust ';
    case 2
        text_str = 'Fear ';
    case 3
        text_str = 'Happiness ';
    case 4
        text_str = 'Neutral ';
    case 5
        text_str = 'Sadness ';
    case 6
        text_str = 'Surprise ';
end

position = [0 0;0 0;0 0]; 
box_color = {'red'};

RGB = insertText(ImgTest,position,[text_str,num2str(accuracy)],'FontSize',18,'BoxColor',...
    box_color,'BoxOpacity',0.4,'TextColor','white');
figure(1); imshow(RGB)