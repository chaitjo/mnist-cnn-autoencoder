function cnnPreprocess()
    %'..\Images_Data_Clipped'
    dataTrainstore = imageDatastore( '..\Images_Data_Clipped\Train\*','LabelSource','foldernames'); %Modify the folder location in this line
    dataTeststore = imageDatastore( '..\Images_Data_Clipped\Train\*','LabelSource','foldernames'); %Modify the folder location in this line
    
    dataTrainstore = shuffle(dataTrainstore);
    dataTeststore = shuffle(dataTeststore);
    
    save('dataTeststore.mat', 'dataTeststore'); %imageDatastore
    save('dataTrainstore.mat', 'dataTrainstore'); %imageDatastore

end