import torch
import torch.nn.functional as F
from model import GRU
from data import tokenizer, label_encoder, combine_entity_text
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_path = "model/best_model.pth"  
vocab_size = 30522  # based on your tokenizer's vocab size
model = GRU(vocab_size)
model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
model.eval()

def predict_single_text(entity, text):

    processed_text = combine_entity_text(entity, text)
    encoding = tokenizer(
        processed_text,
        truncation=True,
        padding='max_length',
        max_length=128,
        return_tensors='pt'
    )
    
    with torch.no_grad():
        outputs = model(encoding['input_ids'], encoding['attention_mask'])
        predicted_class = torch.argmax(outputs, dim=1).item()
        probabilities = F.softmax(outputs, dim=1)[0]

    predicted_sentiment = label_encoder.inverse_transform([predicted_class])[0]
    confidence = probabilities[predicted_class].item()

    print(f"\nEntity: {entity}")
    print(f"Text: {text}")
    print(f"Result: {predicted_sentiment}")
    print(f"Confidence: {confidence:.4f}")

    print("\nThe probabilities of all classes:")
    all_labels = label_encoder.classes_

    label_prob_pairs = list(zip(all_labels, probabilities))
    label_prob_pairs.sort(key=lambda x: x[1], reverse=True)
    
    for label, prob in label_prob_pairs:
        marker = " ✓" if label == predicted_sentiment else ""
        print(f"  {label}: {prob:.4f}{marker}")
    
    return predicted_sentiment

if __name__ == "__main__":
    
    while True:
        entity = input("\nEnter related entity: ")
        text = input("Enter a text to analyze sentiment: ")
        print("Or enter 'exit' to quit")

        if text.lower() == 'exit':
            break

        predict_single_text(entity, text)