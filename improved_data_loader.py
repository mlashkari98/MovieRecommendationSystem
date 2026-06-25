import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import random


class DataLoader:
    '''
    Data Loader class which makes dataset for training / knowledge graph dictionary
    '''
    def __init__(self, data):
        self.cfg = {
            'movie': {
                'item2id_path': 'data/movie/item_index2entity_id.txt',
                'kg_path': 'data/movie/kg.txt',
                'rating_path': 'data/movie/ratings.csv',
                'rating_sep': ',',
                'threshold': 4.5
            },
            'music': {
                'item2id_path': 'data/music/item_index2entity_id.txt',
                'kg_path': 'data/music/kg.txt',
                'rating_path': 'data/music/user_artists.dat',
                'rating_sep': '\t',
                'threshold': 0.0
            }
        }
        self.data = data
        
        df_item2id = pd.read_csv(self.cfg[data]['item2id_path'], sep='\t', header=None, names=['item','id'])
        df_kg = pd.read_csv(self.cfg[data]['kg_path'], sep='\t', header=None, names=['head','relation','tail'])
        df_rating = pd.read_csv(self.cfg[data]['rating_path'], sep=self.cfg[data]['rating_sep'], names=['userID', 'itemID', 'rating', 'timestamp'], skiprows=1)
        
        DATA_PATH = '../Data/ml-1m/'
        USERS_FILE_PATH = DATA_PATH + 'users.dat'
        self.users_dataframe = pd.read_table(USERS_FILE_PATH, delimiter='::', header=None, engine='python', usecols=[0,1,2,3])
        self.users_dataframe.columns = ['userID','Gender', 'Age', 'Occupation']

        # df_rating['itemID'] and df_item2id['item'] both represents old entity ID
        df_rating = df_rating[df_rating['itemID'].isin(df_item2id['item'])]
        df_rating.reset_index(inplace=True, drop=True)
        
        self.df_item2id = df_item2id
        self.df_kg = df_kg
        self.df_rating = df_rating
        
        self.user_encoder = LabelEncoder()
        self.entity_encoder = LabelEncoder()
        self.relation_encoder = LabelEncoder()
        self.gender_encoder = LabelEncoder()
        self._encoding()
        
    def _encoding(self):
        '''
        Fit each label encoder and encode knowledge graph
        '''
        self.user_encoder.fit(self.df_rating['userID'])
        # df_item2id['id'] and df_kg[['head', 'tail']] represents new entity ID
        self.entity_encoder.fit(pd.concat([self.df_item2id['id'], self.df_kg['head'], self.df_kg['tail']]))
        self.relation_encoder.fit(self.df_kg['relation'])
        
        # encode df_kg
        self.df_kg['head'] = self.entity_encoder.transform(self.df_kg['head'])
        self.df_kg['tail'] = self.entity_encoder.transform(self.df_kg['tail'])
        self.df_kg['relation'] = self.relation_encoder.transform(self.df_kg['relation'])

    def _build_dataset(self):
        '''
        Build dataset for training (rating data)
        It contains negative sampling process
        '''
        print('Build dataset dataframe ...', end=' ')
        # users_dataframe update
        self.users_dataframe['Gender'] = self.gender_encoder.fit_transform(self.users_dataframe['Gender'])
        self.users_dataframe['userID'] = self.user_encoder.transform(self.users_dataframe['userID'])
        self.users_dataframe = pd.get_dummies(
            data=self.users_dataframe,
            columns=['Occupation']
        )
        # df dataframe update
        df_dataset = pd.DataFrame()
        df_dataset['userID'] = self.user_encoder.transform(self.df_rating['userID'])
        item2id_dict = dict(zip(self.df_item2id['item'], self.df_item2id['id']))
        self.df_rating['itemID'] = self.df_rating['itemID'].apply(lambda x: item2id_dict[x])
        df_dataset['itemID'] = self.entity_encoder.transform(self.df_rating['itemID'])
        df_dataset['rating'] = self.df_rating['rating']

        df_dataset.reset_index(inplace=True, drop=True)
        print('Done')
        return df_dataset, self.users_dataframe
        
        
    def _construct_kg(self):
        '''
        Construct knowledge graph
        Knowledge graph is dictionary form
        'head': [(relation, tail), ...]
        '''
        print('Construct knowledge graph ...', end=' ')
        kg = dict()
        for i in range(len(self.df_kg)):
            head = self.df_kg.iloc[i]['head']
            relation = self.df_kg.iloc[i]['relation']
            tail = self.df_kg.iloc[i]['tail']
            if head in kg:
                kg[head].append((relation, tail))
            else:
                kg[head] = [(relation, tail)]
            if tail in kg:
                kg[tail].append((relation, head))
            else:
                kg[tail] = [(relation, head)]
        print('Done')
        return kg
        
    def load_dataset(self):
        return self._build_dataset()

    def load_kg(self):
        return self._construct_kg()
    
    def get_encoders(self):
        return (self.user_encoder, self.entity_encoder, self.relation_encoder)
    
    def get_num(self):
        return (len(self.user_encoder.classes_), len(self.entity_encoder.classes_), len(self.relation_encoder.classes_))
