# Next-Word Prediction Using LSTM and Word Embeddings

This project demonstrates a basic language model for **next-word prediction** using TensorFlow and Keras. The model is trained on a small educational text and uses tokenization, word embeddings, LSTM, and softmax output to predict the next likely word given a sequence.

---

## 📒 Project Highlights

* Tokenization and vocabulary generation using Keras Tokenizer
* Sequence padding and input-output preparation
* Model architecture: Embedding → LSTM → Dense (softmax)
* One-hot encoding of output classes
* Prediction of next word given a starting phrase

---

## 🚀 Techniques Used

| Technique              | Purpose                                       |
| ---------------------- | --------------------------------------------- |
| Tokenization           | Convert text into integer sequences           |
| Padding                | Standardize sequence lengths                  |
| Embedding Layer        | Represent words as dense vectors              |
| LSTM                   | Learn temporal dependencies in word sequences |
| Softmax Classification | Predict next word from vocabulary             |
| One-hot Encoding       | Encode output word classes                    |

---

## 💡 Technologies

* Python
* TensorFlow / Keras
* NumPy

---

## 📆 Dataset

The model is trained on a short, self-contained paragraph about data and analytics:

> "Data plays a vital role in our everyday life... this data needs to be cleaned before considering for analysis."

This serves as a minimal training corpus for testing sequence-based predictions.


---

## 🔄 Example Inference

```python
text_1 = "Data is a vital"
token_text = tokenizer.texts_to_sequences([text_1])[0]
padded_text = pad_sequences([token_text], maxlen=33, padding='pre')
pred = model.predict(padded_text)
word_id = np.argmax(pred)

# Reverse lookup
for word, index in tokenizer.word_index.items():
    if index == word_id:
        print(word)
```

---

## 🎯 Results

* Predicts next most probable word based on LSTM-learned context
* Works on small corpus, scalable to larger datasets


---

## 🧠 Skills Gained

* NLP preprocessing (tokenization, padding, encoding)
* Building and training an LSTM model
* Understanding sequence modeling and softmax prediction
* Performing inference and decoding model outputs

---


