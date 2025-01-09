import streamlit as st

from src.generator import WordGenerator
from src.saveload import load

MODEL_PATH = "model.pt"
N_OUTPUTS = 5

st.set_page_config(page_title="HUN Generator")
st.title("Magyar helységnév-generátor")

topcol1, topcol2 = st.columns(2)

with topcol1:
    st.text("Készíts új helységneveket néhány egyszerű beállítással.")
    st.text("Először add meg, hogyan kezdődjenek a nevek.")
    st.text("Utána állítsd be a fantázia mértékét (minél nagyobb, annál változatosabbak a megoldások).")
    st.text("(A haladó beállítások a nagy nyelvi modellek tipikus beállításaihoz hasonlók.)")

with topcol2:
    st.image("HU_counties_blank.svg.png")


@st.cache_resource
def load_model_vocab(path):
    model, vocab = load(path)
    return model, vocab

model, vocab = load_model_vocab(MODEL_PATH)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Beállítások")
    prefix = st.text_input("Névkezdet:", "Hi").lower()
    min_len, max_len = 10, 30
    temperature = st.slider(
        "Fantázia:", value=1.0, min_value=0.0, max_value=2.0, step=0.1
    )
    with st.expander("Haladó beállítások"):
        topk = st.slider(
            "Top k:", value=10, min_value=0, max_value=20
        )
        topp = st.slider(
            "Top p:", value=0.8, min_value=0.0, max_value=1.0
        )

generator = WordGenerator(
    model,
    vocab,
    block_file="data/helysegnevek.txt",
    top_k=topk,
    top_p=topp,
    temperature=temperature,
)

invalid_chars = generator.validate_prefix(prefix)

with col2:
    st.subheader("Kitalált nevek")
    if st.button("Készíts még 5-öt!", use_container_width=True):
        if len(invalid_chars) == 0:
            try:
                for _ in range(N_OUTPUTS):
                        generated = generator.generate_word(prefix=prefix)
                        st.text(generated)
            except RecursionError:
                st.write(
                    ":red[Nem megy. :( Próbáld más beállításokkal. Esetleg]"
                    ":red[ a névkezdet túlságosan hasonlít létező névre?]"
                )
        else:
            invalid_chars_str = ", ".join(invalid_chars)
            st.write(
                f"Módosíts a kezdeten, mert "
                f"ilyen karakter nincs magyar helységnévben: "
                f":red[{invalid_chars_str}]"
            )
