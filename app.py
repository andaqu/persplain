from transformers import RobertaForSequenceClassification, RobertaTokenizer
from simpletransformers.classification import MultiLabelClassificationModel
from transformers_interpret import MultiLabelClassificationExplainer
import streamlit as st
import pandas as pd
import numpy as np

traits = ["Openness to Experience", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism" ]

short_traits = ["o", "c", "e", "a", "n"]

@st.experimental_memo
def load_explainer():

    print("Loading model...")

    tokenizer = RobertaTokenizer.from_pretrained("andaqu/roBERTa-pers")
    model = RobertaForSequenceClassification.from_pretrained("andaqu/roBERTa-pers", problem_type="multi_label_classification")

    explainer = MultiLabelClassificationExplainer(model, tokenizer)

    try:
        model.to('cuda')
        if next(model.parameters()).is_cuda:
            print("Using GPU for inference!")
    except:
        print("GPU not available, using CPU instead.")

    print("Model loaded!")

    return explainer

@st.experimental_memo
def explain(text, _explainer):
    
    attributions = _explainer(text)

    preds = {label: pred_prob.item() for pred_prob, label in zip(_explainer.pred_probs_list, _explainer.labels)}

    attributions_html = {trait : attributions_to_html(attributions[trait]) for trait in attributions}

    return {"preds": preds, "word_attributions_html": attributions_html }

def attributions_to_html(attributions):
  html = ""
  for word, attr in attributions:
      
    if word in ["<s>", "</s>"]:
        continue
    
    attr = round(attr, 2)
    abs_attr = abs(attr)

    color = "rgba(255,255,255,0)"
    if attr > 0: color = f"rgba(0,255,0,{abs_attr})"
    elif attr < 0: color = f"rgba(255,0,0,{abs_attr})"
      
    html += f'<span style="background-color: {color}" title="{str(attr)}">{word}</span> '
  
  return html


if "text" in st.session_state: text = st.session_state.text
else: st.session_state.text = ""

if "explanation" in st.session_state: preds = st.session_state.explanation
else: st.session_state.explanation = {"preds": {}, "word_attributions_html": ""}


def main():

    st.title("Text to Personality Explainer 📊")
    text = ""

    explainer = load_explainer()

    text = st.text_area(label="Input text here...", value="I enjoy meeting people and working hard!")

    show_prediction = st.button("Predict Traits")

    st.session_state.text = text

    if show_prediction and st.session_state.text:
            
            explanation = explain(text, explainer)
            st.session_state.explanation = explanation

    if len(st.session_state.explanation["preds"]) > 0:

        st.write("## Predicted Traits")

        prediction = ["YES" if st.session_state.explanation["preds"][x] > 0.5 else "NO" for x in st.session_state.explanation["preds"]]
        probability = [str(round(st.session_state.explanation["preds"][x]*100)) + "%" for x in st.session_state.explanation["preds"]]
        
        st.table(pd.DataFrame([prediction, probability], columns=traits, index=["Predicted Traits", "Probability"]))

        st.write("## Explanation")
        # Show five buttons, horizontally, one for each trait
        cols = st.columns(5)

        for i in range(5):
            button = cols[i].button(traits[i])
        
            if button: st.markdown(st.session_state.explanation["word_attributions_html"][short_traits[i]], unsafe_allow_html=True)



if __name__ == "__main__":
    main()
