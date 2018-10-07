#Transforming 386 class to 193 class before cclassification

import pandas as pd

data =  pd.read_csv('features.csv')
cl = sorted(list(set(data['label'])))
label = {
    c : i for i,c in enumerate(cl)

}

data['label'] = data['label'].replace(label)
data.to_csv('final_features.csv',index = False)