from flask import Flask, request, jsonify
import torch
import torch.nn as nn
import tensorflow as tf
import joblib
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from train_model_pytorch import ExpenseClassifier  # Import model class

app = Flask(__name__)

# ✅ Load Required Files (Check for Missing Files)
required_files = ["label_encoder.pkl", "tokenizer.pkl", "vocab.pkl", "max_length.pkl", 
                  "expense_classifier_tf.h5", "expense_classifier_pytorch.pth"]

for file in required_files:
    try:
        open(file).close()
    except FileNotFoundError:
        raise FileNotFoundError(f"❌ Missing required file: {file}. Ensure training was completed.")

# ✅ Load Label Encoder
label_encoder = joblib.load("label_encoder.pkl")

# ✅ Load TensorFlow Model
tf_model = tf.keras.models.load_model("expense_classifier_tf.h5")

# ✅ Load Tokenizer & Max Length
tokenizer = joblib.load("tokenizer.pkl")
max_length = joblib.load("max_length.pkl")

# ✅ Load Vocabulary & PyTorch Model
vocab = joblib.load("vocab.pkl")
vocab_size = len(vocab) + 1  # Ensure correct vocab size

# ✅ Load PyTorch Model with Updated Architecture
num_classes = len(label_encoder.classes_)
embed_dim = 64  # Match training

pytorch_model = ExpenseClassifier(vocab_size=vocab_size, embed_dim=embed_dim, num_classes=num_classes)
pytorch_model.load_state_dict(torch.load("expense_classifier_pytorch.pth", map_location=torch.device("cpu")))
pytorch_model.eval()

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    description = data.get("description", "").strip()

    if not description:
        return jsonify({"error": "Description is required"}), 400

    # ✅ Preprocess input for TensorFlow model
    tf_sequence = tokenizer.texts_to_sequences([description])
    tf_padded_sequence = pad_sequences(tf_sequence, maxlen=max_length, padding="post")

    # ✅ Make prediction with TensorFlow
    tf_prediction = tf_model.predict(tf_padded_sequence)
    tf_predicted_label = label_encoder.inverse_transform([np.argmax(tf_prediction)])

    # ✅ Convert text to sequence for PyTorch
    pytorch_sequence = [vocab.get(word, 0) for word in description.lower().split()]
    pytorch_padded = pytorch_sequence + [0] * (max_length - len(pytorch_sequence))  # Pad sequence
    pytorch_input = torch.tensor([pytorch_padded], dtype=torch.long)

    with torch.no_grad():
        pytorch_prediction = pytorch_model(pytorch_input)
        pytorch_pred = torch.argmax(pytorch_prediction, dim=1).cpu().numpy()  # Convert to NumPy
        pytorch_predicted_label = label_encoder.inverse_transform(pytorch_pred)

    # ✅ Debug Logging
    print(f"🔹 Input: {description}")
    print(f"➡ TensorFlow Prediction: {tf_predicted_label[0]}")
    print(f"➡ PyTorch Prediction: {pytorch_predicted_label[0]}")

    return jsonify({
        "tensorflow_category": tf_predicted_label[0],
        "pytorch_category": pytorch_predicted_label[0]
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
