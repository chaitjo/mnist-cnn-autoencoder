%autoencoderPreprocess('..\Images_Data_Clipped');

load 'dataTest.mat';
load 'dataTrain.mat';
load 'labelsTest.mat';
load 'labelsTrain.mat';


hiddenSize1 = 150;

autoenc1 = trainAutoencoder(dataTrain, hiddenSize1, ...
    'MaxEpochs',200, ...
    'SparsityRegularization',2, ...
    'SparsityProportion',0.2);

figure(), plotWeights(autoenc1);

reconstructed = predict(autoenc1, dataTest);



mseError = 0;
for i = 1:numel(dataTest)
    mseError = mseError + mse(double(dataTest{1, i}) - reconstructed{1, i});
end

mseError = mseError/i;
disp('L1 MSE: ');
disp(mseError);


feat1 = encode(autoenc1, dataTrain);

hiddenSize2 = 100;

autoenc2 = trainAutoencoder(feat1, hiddenSize2, ...
    'MaxEpochs', 200, ...
    'SparsityRegularization', 2, ...
    'SparsityProportion', 0.2);

figure(), plotWeights(autoenc2);


% calculate L2 MSE
feat1Test = encode(autoenc1, dataTest);
reconstructedFeat1 = predict(autoenc2, feat1Test);
reconstructed = decode(autoenc1, reconstructedFeat1);

mseError = 0;
for i = 1:size(feat1Test, 1)
    mseError = mseError + mse(double(dataTest{1, i}) - reconstructed{1, i});
end

mseError = mseError/i;
disp('L2 MSE: ');
disp(mseError);

feat2 = encode(autoenc2,feat1);


softnet = trainSoftmaxLayer(feat2, labelsTrain, 'MaxEpochs',200);

% Build stacked autoencoder
deepnet = stack(autoenc1, autoenc2, softnet);


% Reshape test images into vectors
xTest = zeros(28*28,numel(dataTest));
for i = 1:numel(dataTest)
    xTest(:,i) = dataTest{i}(:);
end

y = deepnet(xTest);
plotconfusion(labelsTest,y);


figure;
for i = 1:20
    subplot(4,5,i);
    imshow(dataTest{i});
end

figure;
for i = 1:20
    subplot(4,5,i);
    imshow(reconstructed{i});
end

