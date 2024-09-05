# IHEPC
IHEPC: Improved hemolytic peptide prediction with comprehensive hemolytic peptide sequences

# Description
This is the source code of IHEPC, a machine-learning predictor for Hemolytic peptides. The trained models are also included in this package, facilitating prediction on a given data set.

# Installation
Requirements:
* Python = 3.8, pycaret[full] = 2.3.10

Packages
* Install required packages using `pip install -r requirements.txt`

# Usage
Modify main_predict.py for your data set in fasta format
* Input file
  * One or more files in Fasta format (described below)
  
* output file
  * binary_vector.csv -- The prediction output in binary format (1 for positive and 0 for negative)
    
    ![vector](https://github.com/csh07/HEPAD/assets/145912636/853a2868-444a-4e67-99c5-713c16507773)

  * probability.csv -- The prediction probability estimate
    
    ![prob](https://github.com/csh07/HEPAD/assets/145912636/9fbba106-faf9-4dc4-9455-775f32463b90)

When dataset = 'Hmp1', the program will use models trained on Hmp1, corresponding features, and their normalization scaler to process data and perform prediction.

When dataset = 'Hmp2', the program will use models trained on Hmp2, corresponding features, and their normalization scaler to process data and perform prediction.

When dataset = 'Hmp3', the program will use models trained on Hmp3, corresponding features, and their normalization scaler to process data and perform prediction.

When dataset = 'Hmpm', the program will use models trained on Hmpm, corresponding features, and their normalization scaler to process data and perform prediction.
```py
# If you want to use a different model, you can change the dataset
dataset = 'Hmpm'
if dataset == 'Hmp1':
    model_use = '1'
elif dataset == 'Hmp2':
    model_use = '2'
elif dataset == 'Hmp3':
    model_use = '3'
elif dataset == 'Hmpm':
    model_use = '4'
```

```py
# Path setting
pathDict = {'paramPath': f'../data/param/{dataset}/',  # This path should have featureTypeDict.pkl and robust.pkl
            'saveCsvPath': '../data/mlData/new_data/',  # Your encoded data will save in this path
            'modelPath': f'../data/finalModel/{dataset}/',  # This path should have 'rbfsvm', 'lightgbm', 'gbc' models. ex: gbc_final.pkl
            'outputPath': f'../data/output/{dataset}/'}  # Your prediction will save in this path
```


Specify one or more fasta files in the 'inputPathList' parameter. Sequences from these fasta files will be concatenated for prediction, and prediction results will be written to the default output files, binary_vector.csv and probability.csv.

```py
# Input your FASTA file, the example file can find in data/mlData/Hmp1/test_neg.FASTA
inputPathList = [f'../data/mlData/{dataset}/test_neg.FASTA', f'../data/mlData/{dataset}/test_pos.FASTA']
```

Here is the code snippet in main_predict.py. We already set the parameters and the program is ready to be excecuted.

```py
encapObj = HEPAD_Predict(model_use=model_use, pathDict=pathDict)
encapObj.loadData(inputDataList=inputPathList)
encapObj.featureEncode()
encapObj.doPredict()
```
