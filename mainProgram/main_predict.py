from userPackage.Package_IHPP import IHPP_Predict

# If you want to use a different model, you can change the dataset
dataset = 'DS2'
if dataset == 'DS1':
    model_use = '1'
elif dataset == 'DS2':
    model_use = '2'


# Path setting
pathDict = {'paramPath': f'../data/param/{dataset}/',  # This path should have featureTypeDict.pkl and robust.pkl
            'saveCsvPath': '../data/mlData/new_data/',  # Your encoded data will save in this path
            'modelPath': f'../data/finalModel/{dataset}/',  # This path should have 'rbfsvm', 'lightgbm', 'gbc' models. ex: gbc_final.pkl
            'outputPath': f'../data/output/{dataset}/'}  # Your prediction will save in this path

# Input your FASTA file, the example file can find in data/mlData/DS1/test_neg.FASTA
inputPathList = [f'../data/mlData/{dataset}/test_neg.FASTA', f'../data/mlData/{dataset}/test_pos.FASTA']

encapObj = IHPP_Predict(model_use=model_use, pathDict=pathDict)
encapObj.loadData(inputDataList=inputPathList)
encapObj.featureEncode()
encapObj.doPredict()


