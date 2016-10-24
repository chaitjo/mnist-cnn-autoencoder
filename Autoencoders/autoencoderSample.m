%autoencoderPreprocess('..\Images_Data_Clipped');

load 'dataTest.mat';
load 'dataTrain.mat';

%Take 100 samples for training and 20 for testing
dataTestSubset = dataTest(1, 1:20);
dataTrainSubset = dataTrain(1, 1:100);

hiddenSize1 = 100;

autoenc1 = trainAutoencoder(dataTrainSubset,hiddenSize1);

figure(), plotWeights(autoenc1);

reconstructed = predict(autoenc1, dataTestSubset);

mseError = 0;
for i = 1:numel(dataTestSubset)
    mseError = mseError + mse(double(dataTestSubset{1, i}) - reconstructed{1, i});
end

mseError = mseError/i;

figure;
for i = 1:20
    subplot(4,5,i);
    imshow(dataTestSubset{i});
end

figure;
for i = 1:20
    subplot(4,5,i);
    imshow(reconstructed{i});
end

