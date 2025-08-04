import re
import string
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from transformers import AutoTokenizer

stop_words = set(stopwords.words('english')) 
lemmatizer = WordNetLemmatizer() 

train_df = pd.read_csv('data/twitter_training.csv')
test_df = pd.read_csv('data/twitter_validation.csv')
columns = ['Tweet ID', 'entity', 'sentiment', 'Tweet content']

train_df.columns = columns
test_df.columns = columns

train_df.drop(columns=['Tweet ID'], inplace=True)
train_df.drop_duplicates(inplace=True)
train_df.dropna(subset=['Tweet content'], inplace=True)

test_df.drop(columns=['Tweet ID'], inplace=True)
test_df.drop_duplicates(inplace=True)
test_df.dropna(subset=['Tweet content'], inplace=True)

train_data, val_data = train_test_split(
    train_df,
    test_size=0.2,
    random_state=42,
    stratify=train_df['sentiment']
)

def clean_text(text):
    if isinstance(text, str):
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'\@\w+|\#\w+', '', text)
        text = text.lower()
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = ' '.join(lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words)
        return text
    else:
        return ""

def combine_entity_text(entity, text):
    return f"[ENTITY] {entity.lower()} [TEXT] {text}"

train_data['Tweet content'] = train_data['Tweet content'].apply(clean_text)
val_data['Tweet content'] = val_data['Tweet content'].apply(clean_text)
test_df['Tweet content'] = test_df['Tweet content'].apply(clean_text)

train_data['combined_text'] = train_data.apply(lambda row: combine_entity_text(row['entity'], row['Tweet content']), axis=1)
val_data['combined_text'] = val_data.apply(lambda row: combine_entity_text(row['entity'], row['Tweet content']), axis=1)
test_df['combined_text'] = test_df.apply(lambda row: combine_entity_text(row['entity'], row['Tweet content']), axis=1)

label_encoder = LabelEncoder()
train_data['sentiment_encoded'] = label_encoder.fit_transform(train_data['sentiment'])
val_data['sentiment_encoded'] = label_encoder.transform(val_data['sentiment'])
test_df['sentiment_encoded'] = label_encoder.transform(test_df['sentiment'])

class TwitterDataset(Dataset):
    def __init__(self, texts, labels, tokenizer=None, max_length=128, vectorizer=None, use_tfidf=False):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.vectorizer = vectorizer
        self.use_tfidf = use_tfidf
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts.iloc[idx])
        label = self.labels.iloc[idx]
        
        # text_vector = self.vectorizer.transform([text]).toarray()[0]
        # return {
        #         'input_features': torch.tensor(text_vector, dtype=torch.float32),
        #         'label': torch.tensor(label, dtype=torch.long)
        # }

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

tokenizer = None
vectorizer = None

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
# vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')

def create_dataloaders(batch_size=32, max_length=128):
    
    train_dataset = TwitterDataset(
        train_data['combined_text'], 
        train_data['sentiment_encoded'], 
        tokenizer, 
        max_length
    )
    val_dataset = TwitterDataset(
        val_data['combined_text'], 
        val_data['sentiment_encoded'], 
        tokenizer, 
        max_length
    )
    test_dataset = TwitterDataset(
        test_df['combined_text'], 
        test_df['sentiment_encoded'], 
        tokenizer, 
        max_length
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0  
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    
    return train_loader, val_loader, test_loader

