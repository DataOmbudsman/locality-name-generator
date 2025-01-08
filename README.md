A character-level LSTM model that generates Hungarian locality names.

# Training

`python -m src.train --input data/helysegnevek.txt --output model.pt`

# Inference

`python -m src.infer --prefix "he"  --model model.pt`

# Streamlit app

`streamlit run app.py`
