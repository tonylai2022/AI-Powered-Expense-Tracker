import pandas as pd
import os
import tensorflow as tf
import joblib
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D, Dense
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# ✅ Disable OneDNN warnings
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# ✅ Load dataset
file_path = "mock_expense_data.xlsx"
df = pd.read_excel(file_path)

# ✅ Ensure correct column names (for safety)
df.columns = df.columns.str.lower()

# ✅ Validate necessary columns exist
if "description" not in df.columns or "category" not in df.columns:
    raise ValueError("❌ Error: Missing 'description' or 'category' column in dataset!")

# ✅ Drop NaN values to avoid issues
df.dropna(subset=["description", "category"], inplace=True)

# ✅ Encode labels
label_encoder = LabelEncoder()
df["category"] = label_encoder.fit_transform(df["category"])

# ✅ Save Label Encoder for later use
joblib.dump(label_encoder, "label_encoder.pkl")

# ✅ Tokenize text
tokenizer = Tokenizer(num_words=None, oov_token="<OOV>")  # Set dynamically
tokenizer.fit_on_texts(df["description"])
num_words = len(tokenizer.word_index) + 1  # Get actual vocab size
tokenizer.num_words = num_words  # Ensure correct num_words is set

# ✅ Save tokenizer
joblib.dump(tokenizer, "tokenizer.pkl")

# ✅ Convert text to sequences
sequences = tokenizer.texts_to_sequences(df["description"])

# ✅ Pad sequences
max_length = max(len(seq) for seq in sequences)
X = pad_sequences(sequences, maxlen=max_length, padding="post")
y = df["category"].values

# ✅ Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ Define Neural Network Model
model = Sequential([
    Embedding(num_words, 32, input_length=max_length),
    GlobalAveragePooling1D(),
    Dense(16, activation="relu"),
    Dense(len(label_encoder.classes_), activation="softmax")
])

model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# ✅ Train Model
model.fit(X_train, y_train, epochs=5, batch_size=16, validation_data=(X_test, y_test))

# ✅ Save Model
model.save("expense_classifier_tf.h5")

# ✅ Save Max Sequence Length for Consistent Input Processing
joblib.dump(max_length, "max_length.pkl")

print("✅ TensorFlow model, tokenizer, and label encoder saved!")

# ✅ Debugging: Print Sample Tokenized Data
print("\n📌 Debugging Info:")
print(f"- Vocabulary Size: {num_words}")
print(f"- Max Sequence Length: {max_length}")
print(f"- Sample Tokenized Input: {tokenizer.texts_to_sequences(['Bought Coffee'])}")
print(f"- Label Mapping: {dict(enumerate(label_encoder.classes_))}")


