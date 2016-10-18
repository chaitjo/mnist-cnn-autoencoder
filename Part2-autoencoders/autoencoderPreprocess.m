function autoencoderPreprocess()
       %..\Images_Data_Clipped
    dataTrainObject = imageDatastore('..\Images_Data_Clipped\Train\*','LabelSource','foldernames'); %Modify the folder location in this line
    dataTestObject = imageDatastore( '..\Images_Data_Clipped\Test\*','LabelSource','foldernames');  %Modify the folder location in this line

    labelsTest = dataTestObject.Labels;
    labelsTest = uint8(labelsTest);
    labelsTest = labelsTest - 1;
    labelsTest(labelsTest == 0) = 10;
    labelsTestSoftmax = zeros(10, size(labelsTest, 1));

    dataTest = cell(1, numel(labelsTest)); 

    for  i = (1: size(labelsTest, 1))
        %disp(dataTrainObject.Files{i});
        filename = dataTestObject.Files{i};
        file = imread(filename);
        dataTest{1, i} = file;
    end



    labelsTrain = dataTrainObject.Labels;
    labelsTrain = uint8(labelsTrain);
    labelsTrain = labelsTrain - 1;
    labelsTrain(labelsTrain == 0) = 10;
    labelsTrainSoftmax = zeros(10, size(labelsTrain, 1));

    dataTrain = cell(1, numel(labelsTrain));


    for  i = (1: size(labelsTrain, 1))
        %disp(dataTrainObject.Files{i});
        filename = dataTrainObject.Files{i};
        file = imread(filename);
        %disp(size(file(:)));
        dataTrain{1, i} = file;
    end

    randTest = randperm(size(dataTest, 2));
    dataTest = dataTest(:, randTest);
    labelsTest = labelsTest(randTest);

    for  i = (1: size(labelsTest, 1))
        labelsTestSoftmax(uint8(labelsTest(i)), i) = 1;
    end

    randTrain = randperm(size(dataTrain, 2));
    dataTrain = dataTrain( :, randTrain);
    labelsTrain = labelsTrain(randTrain);

    for  i = (1: size(labelsTrain, 1))
        labelsTrainSoftmax(uint8(labelsTrain(i)), i) = 1;
    end

    labelsTest = labelsTestSoftmax;
    labelsTrain = labelsTrainSoftmax;

    save('dataTest.mat', 'dataTest');
    save('labelsTest.mat', 'labelsTest');
    save('dataTrain.mat', 'dataTrain');
    save('labelsTrain.mat', 'labelsTrain');

end