%% Import and segment data
clc, clear

xTest2 = loadmnist2();
[xTrain, tTrain, xValid, tValid, xTest, tTest] = LoadMNIST(3);
 %% Construct & Train Network
 clc
 
 % Define parameters
 imageInputSize = [28 28 1];
 numberOfClasses = 10; % 10 digits

 % Define network layout
 layers = [
    imageInputLayer(imageInputSize,'Name','Input')
    convolution2dLayer(3,20,'Padding','same')
    batchNormalizationLayer
    reluLayer

    maxPooling2dLayer(3,"Stride",2,Padding="same")

    convolution2dLayer(3,30,'Padding','same','Stride',2)
    batchNormalizationLayer
    reluLayer

    %maxPooling2dLayer(4,"Stride",3,Padding="same")

    fullyConnectedLayer(numberOfClasses)
    softmaxLayer
    classificationLayer];
 lGraph = layerGraph;
 lGraph = addLayers(lGraph,layers);
 %plot(lGraph)

 % Train the network using SGDM
 options = trainingOptions("sgdm", ...
     'ExecutionEnvironment','gpu', ...
     "InitialLearnRate",0.002, ...
     "MaxEpochs",10, ...
     "MiniBatchSize",128, ...
     'Shuffle',"every-epoch", ...
     'Plots',"training-progress", ...
     'L2Regularization',0.0005);

net = trainNetwork(xTrain, tTrain, layers, options);

%% Apply network to data
% Use network to predict training data, calculating accuracy and error
predictTrainData = net.classify(xTrain);
accuracyTrain = sum(predictTrainData==tTrain)/numel(tTrain)*100; % percent
classErrorTrain = (100 - accuracyTrain)

% Do the same for the validation data
predictValData = net.classify(xValid);
accuracyVal = sum(predictValData==tValid)/numel(tValid)*100;
classErrorVal = (100 - accuracyVal)

% And for the test data
predictTestData = net.classify(xTest);
accuracyTest = sum(predictTestData==tTest)/numel(tTest)*100;
classErrorTest = (100 - accuracyTest)

predictTestData2 = net.classify(xTest2);

writematrix(predictTestData2,'classifications.csv')