'''
data_prep.py

Starting with the csv's, ending with X_train, y_train, X_val, y_val, X_test, y_test
Where X's are feature vectors and y's are classifier integers
'''
__author__='Charlie Guthrie'

#TODO: 
# 1. test and debug train_val_split

def train_val_split(df,samples,val_portion):
    '''
    split dataframe into validation and training sets
    
    args:
        df: data frame to split
        samples: number of samples
        val_portion: fraction (between 0 and 1) of samples to devote to validation
    returns:
        trainDF
        valDF
        
    '''
    assert val_portion<1 and val_portion>0
    assert df.shape[0]>2
    df = shuffle_df(df)
    if samples is not None:
        assert samples<=df.shape[0]
        sampleDF = df.iloc[:samples,:]
    else:
        sampleDF = df
    val_samples = round(samples*val_portion)
    train_samples = samples - val_samples
    trainDF = sampleDF.iloc[:train_samples,:]
    valDF = sampleDF.iloc[train_samples:val_samples,:]

    return trainDF, valDF

def strip_columns(df):
	#low-priority TODO
	'''
	returns only the columns needed for this model
	'''

def merge_data(sets,csv_path,text_path, image_path):
	'''
	merge together the datasets to be used in the model
	args:
		sets: list of datasets to be used
	'''
	#TODO

def main(train_df,test_df):
	'''
	1. run train_val_split on training
	2. strip columns
	3. train tokenizer
	4. convert text data to bag of words matrix
	5. merge datasets
	'''
	#TODO

