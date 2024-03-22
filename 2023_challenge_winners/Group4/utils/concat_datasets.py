import os
import pandas as pd
import glob

def concat_csv(topdir, outdir, filename):
    """
    Concatenate all CSV files in a directory into a single CSV file.
    """
    PATH='../data/train/'

    concat_df = pd.DataFrame()

    for i in os.listdir(PATH):
        if os.path.isdir(PATH+i):
            print(PATH+i)
            for subfolder in os.listdir(PATH+i):
                print(subfolder)
                df = pd.read_csv(PATH+i+'/{0}/{0}_Labels.csv'.format(subfolder))
                df['FileName'] = df['FileName'].apply(lambda x: PATH+i+'/{0}/'.format(subfolder)+x)
                df['fold'] = i
                concat_df = pd.concat([concat_df, df], axis=0)
    
    drop_cols = []
    for col in concat_df.columns:
        if 'Unnamed' in col:
            drop_cols.append(col)
    
    concat_df.drop(drop_cols, axis=1).to_csv('train.csv', index=False)

if __name__ == '__main__':
    concat_csv('data/train/', 'data/', 'train.csv')
    #concat_csv('data/test/', 'data/', 'test.csv')