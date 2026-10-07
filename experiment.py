"""Reproducible experimental experiment; all transforms fit training data only."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score,f1_score,confusion_matrix,classification_report
from sklearn.linear_model import LogisticRegression,LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB,MultinomialNB
from sklearn.svm import SVC
def classification(y,p):
    return {'accuracy':float(accuracy_score(y,p)),'macro_f1':float(f1_score(y,p,average='macro',zero_division=0)),'confusion_matrix':confusion_matrix(y,p).tolist(),'report':classification_report(y,p,output_dict=True,zero_division=0)}
def neural(x,y,vx,vy,tx,regression=False):
    import torch,copy
    from torch import nn
    torch.set_num_threads(2);torch.manual_seed(42)
    x,vx,tx=[torch.tensor(z,dtype=torch.float32) for z in (x,vx,tx)]
    scale=max(float(np.std(y)),1.) if regression else 1.
    mean=float(np.mean(y)) if regression else 0.
    y,vy=[torch.tensor((z-mean)/scale,dtype=torch.float32).reshape(-1,1) if regression else torch.tensor(z,dtype=torch.long) for z in (y,vy)]
    model=nn.Sequential(nn.Linear(x.shape[1],32),nn.ReLU(),nn.Linear(32,16),nn.ReLU(),nn.Linear(16,1 if regression else 2))
    criterion=nn.MSELoss() if regression else nn.CrossEntropyLoss()
    optimizer=torch.optim.Adam(model.parameters(),lr=.003)
    best=float('inf');weights=None;wait=0;history=[]
    for epoch in range(100):
        model.train()
        for batch in torch.randperm(len(x)).split(128):
            optimizer.zero_grad();loss=criterion(model(x[batch]),y[batch]);loss.backward();optimizer.step()
        model.eval()
        with torch.inference_mode(): score=float(criterion(model(vx),vy))
        history.append(score)
        if score<best-1e-5:best=score;weights=copy.deepcopy(model.state_dict());wait=0
        else:wait+=1
        if wait>=12:break
    model.load_state_dict(weights)
    with torch.inference_mode():out=model(tx)
    pred=out.numpy().ravel()*scale+mean if regression else out.argmax(1).numpy()
    return pred,{'epochs':len(history),'best_validation_loss':best,'target_scale':scale,'target_mean':mean}
def experiment(data):
    import re
    from nltk.stem import PorterStemmer
    from sklearn.feature_extraction.text import TfidfVectorizer,ENGLISH_STOP_WORDS
    frame=pd.read_excel(data) if Path(data).suffix.lower()=='.xlsx' else pd.read_csv(data)
    frame=frame.dropna(subset=['text','airline_sentiment']).copy()
    stemmer=PorterStemmer()
    def clean(text):
        text=re.sub(r'http\S+|www\S+|@\w+|#\w+',' ',str(text).lower())
        return ' '.join(stemmer.stem(w) for w in re.findall(r'[a-z]+',text) if w not in ENGLISH_STOP_WORDS)
    frame['clean']=frame['text'].map(clean);frame=frame[frame['clean'].str.len()>0]
    conflicts=frame.groupby('clean')['airline_sentiment'].nunique();frame=frame[~frame['clean'].isin(conflicts[conflicts>1].index)].drop_duplicates('clean')
    a,b,ya,yb=train_test_split(frame['clean'],frame['airline_sentiment'],test_size=.3,random_state=42,stratify=frame['airline_sentiment'])
    assert not set(a)&set(b)
    results={}
    for name,model in [('DecisionTree',DecisionTreeClassifier(random_state=42)),('NaiveBayes',MultinomialNB()),('KNN',KNeighborsClassifier(n_neighbors=7))]:
        pipeline=make_pipeline(TfidfVectorizer(max_features=15000),model);pipeline.fit(a,ya);results[name]=classification(yb,pipeline.predict(b))
    return {'dataset':'Twitter US Airline Sentiment','seed':42,'train':len(a),'test':len(b),'removed_empty_conflicting_duplicate_rows':int(len(pd.read_excel(data) if Path(data).suffix.lower()=='.xlsx' else pd.read_csv(data))-len(frame)),'models':results,'limitation':'Random held-out tweets; no user/time holdout. Source tweets and user identifiers are excluded.'}
if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--output',type=Path,default=Path('metrics.json'))
    args=parser.parse_args()
    results=experiment(args.data)
    args.output.write_text(json.dumps(results,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(results))
