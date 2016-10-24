%cnnPreprocess('..\Images_Data_Clipped');

load 'dataTeststore.mat';
load 'dataTrainstore.mat';

dataTrainstoreSubset = imageDatastore('..\');
dataTeststoreSubset = imageDatastore('..\');

%Take 100 samples for training and 20 for testing
dataTrainstoreSubset.Files = dataTrainstore.Files(1:100);
dataTeststoreSubset.Files = dataTeststore.Files(1:20);

dataTrainstoreSubset.Labels = dataTrainstore.Labels(1:100);
dataTeststoreSubset.Labels = dataTeststore.Labels(1:20);

imageDim = 28;

layers = [imageInputLayer([imageDim imageDim]), ...
	 convolution2dLayer([9, 9],50), ...
	 averagePooling2dLayer(4), ...
     fullyConnectedLayer(10), ...
	 softmaxLayer(), ...
     classificationLayer()];
 
 options = trainingOptions('sgdm', ... 
            'MaxEpochs', 50,...
            'InitialLearnRate',0.001, ...
            'MiniBatchSize', 16 ...
        );

convnet = trainNetwork(dataTrainstoreSubset,layers,options);


YTest = classify(convnet, dataTeststoreSubset);
TTest = dataTeststoreSubset.Labels;
accuracy = sum(YTest == TTest)/numel(YTest);