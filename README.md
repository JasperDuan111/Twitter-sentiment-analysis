# Twitter Sentiment Analysis Project

## Project Overview

This is a deep learning-based Twitter sentiment analysis project that uses GRU (Gated Recurrent Unit) neural networks for four-class sentiment classification of Twitter text. The test accuracy reaches 0.96. The project supports the following four sentiment categories:

- **Positive** 
- **Negative** 
- **Neutral** 
- **Irrelevant**

## Project Structure

```
Twitter sentiment analysis/
├── data/                          # Data folder
│   ├── twitter_training.csv       # Training dataset
│   └── twitter_validation.csv     # Validation dataset
├── model/                         # Model folder
│   └── best_model.pth            # Best trained model
├── data.py                       # Data processing and loading module
├── ultrakill.png                 # A picture which you can just ignore 
├── model.py                      # Neural network model definition
├── train.py                      # Model training script
├── test.py                       # Single text testing script
├── data.ipynb                    # Jupyter data analysis notebook
├── requirements.txt              # Project dependencies
└── README.md                     # Project documentation
```
## Dataset Description
The data comes from [Kaggle](https://www.kaggle.com/datasets/jp797498e/twitter-entity-sentiment-analysis/data)

### Data Source
- **Training set**: `data/twitter_training.csv` (74,683 records)
- **Validation set**: `data/twitter_validation.csv`

### Data Format
Each CSV file contains the following columns:
- `Tweet ID`: Unique identifier for tweets
- `entity`: Entity name
- `sentiment`: Sentiment label (Positive/Negative/Neutral/Irrelevant)
- `Tweet content`: Tweet content

### Data Preprocessing
- Remove URL links (`http://`, `www.`, `https://`)
- Remove @usernames and #hashtags
- Convert to lowercase
- Remove punctuation
- Remove stopwords
- Lemmatization processing

## Model Architecture

### GRU Model
```python
GRU(
  (embedding): Embedding(30522, 128, padding_idx=0)
  (gru): GRU(128, 128, num_layers=2, batch_first=True, dropout=0.5)
  (dropout): Dropout(p=0.5)
  (fc): Linear(in_features=128, out_features=4)
)
```

### Model Features
- **Embedding Layer**: 128-dimensional word vectors
- **GRU Layer**: 2 layers, hidden dimension 128, dropout=0.5
- **Classification Layer**: Fully connected layer with 4 output classes
- **Attention Mask Support**: Handles variable-length sequences
- **BERT Tokenizer**: Uses pre-trained BERT tokenizer

### Model Parameters
- Vocabulary size: 30,522 (BERT vocabulary)
- Maximum sequence length: 128
- Learning rate: 0.001
- Optimizer: Adam
- Loss function: CrossEntropyLoss

## Usage

### Train Model
```bash
python train.py
```

The training process will:
1. Load and preprocess data
2. Create data loaders
3. Initialize GRU model
4. Train for 10 epochs
5. Save best model to `model/best_model.pth`
6. Evaluate performance on test set

### Single Text Testing
```bash
python test.py
```

#### Interactive Usage:
```
Enter related entity: Ultrakill
Enter a text to analyze sentiment: Amazing masterpiece!
Or enter 'exit' to quit

Entity: Ultrakill
Text: Amazing masterpiece!
Result: Positive
Confidence: 0.9924

The probabilities of all classes:
  Positive: 0.9924 ✓
  Irrelevant: 0.0054
  Neutral: 0.0019
  Negative: 0.0004
```
![Ultrakill](ultrakill.png)
If you are also interested in this game, click [Ultrakill](https://store.steampowered.com/app/1229490/ULTRAKILL/)

### Performance Optimization Tips

1. **Data Parallelism**: Increase DataLoader num_workers
2. **Mixed Precision**: Use torch.cuda.amp for training
3. **Learning Rate Scheduling**: Add learning rate decay strategy

## Extensions

### Possible Improvements
1. **Model Architecture**: Try LSTM, Transformer
2. **Pre-training**: Use BERT, RoBERTa pre-trained models
3. **Ensemble Methods**: Model fusion for better performance