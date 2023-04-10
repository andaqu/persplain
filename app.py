import gradio as gr
from transformers import RobertaForSequenceClassification, RobertaTokenizer
from transformers_interpret import MultiLabelClassificationExplainer
import pandas as pd
from transformers import logging

logging.set_verbosity_warning()

traits = ["Openness to Experience", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism" ]

short_traits = ["o", "c", "e", "a", "n"]

short_to_long = {"o": "Openness to Experience", "c": "Conscientiousness", "e": "Extraversion", "a": "Agreeableness", "n": "Neuroticism" }

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

explainer = load_explainer()

def explain(text, _explainer):
    if text is not None:
        attributions = _explainer(text)
        preds = {label: pred_prob.item() for pred_prob, label in zip(_explainer.pred_probs_list, _explainer.labels)}
        attributions_html = {trait : attributions_to_html(attributions[trait], trait) for trait in attributions}
        return {"preds": preds, "word_attributions_html": attributions_html }
    else:
        return None

def attributions_to_html(attributions, short_trait):
    html = f""
    for word, attr in attributions:
        if word in ["<s>", "</s>"]:
            continue
        attr = round(attr, 2)
        abs_attr = abs(attr)
        color = "rgba(255,255,255,0)"
        if attr > 0: color = f"rgba(0,255,0,{abs_attr})"
        elif attr < 0: color = f"rgba(255,0,0,{abs_attr})"
        html += f'<span style="background-color: {color}" title="{str(attr)}">{word}</span> '
    html += f"<br>"
    return html

def get_predictions(text):
    explanation = explain(text, explainer)

    prediction = ["YES" if explanation["preds"][x] > 0.5 else "NO" for x in explanation["preds"]]
    probability = [str(round(explanation["preds"][x]*100)) + "%" for x in explanation["preds"]]

    result_df = pd.DataFrame(data={"Predicted Traits": prediction, "Probability": probability}, index=traits)

    def color_row(row):
        if row['Predicted Traits'] == 'YES':
            return ['background-color: green']*len(row)
        else:
            return ['background-color: red']*len(row)

    # apply conditional formatting to dataframe
    result_df = result_df.style.apply(color_row, axis=1)

    def render_html(val):
        return val

    explanation_df = pd.DataFrame(data={"Explanation": [explanation["word_attributions_html"][x] for x in short_traits]}, index=traits)

    explanation_df = explanation_df.style.format({'Explanation': render_html})

    return result_df, explanation_df

def text_to_personality_explainer(text):
    result_df, explanation_df = get_predictions(text)

    return "<center>" + result_df.to_html() + "</center>", "<center>" + explanation_df.to_html() + "</center>"

main = gr.Blocks()
text_input = gr.Textbox(placeholder="Enter text here...")
result = gr.outputs.HTML()
explanation = gr.outputs.HTML() 

with main:
    gr.Markdown("# Text to Personality Explainer 📊")
    gr.Markdown("Predict personality traits from text using a RoBERTa model fine-tuned on a Big Five Personality Traits dataset.")
    gr.Markdown("Explanations are given in the form of word attributions, where the color of the word indicates the importance of the word for the prediction. Green words increase the probability of the trait, red words decrease the probability of the trait.")

    gr.Examples(["I love working and meeting people!", "I am a bad person. :(", "I find it challenging to agree with my brother."], fn=text_to_personality_explainer, inputs=text_input, outputs=[result, explanation], cache_examples=False)

    text_input.render()
    text_button = gr.Button("Predict")

    with gr.Tabs():
        with gr.TabItem("Prediction"):
            result.render()
           
        with gr.TabItem("Explanation"):
            explanation.render()

    text_button.click(text_to_personality_explainer, inputs=text_input, outputs=[result, explanation])

main.launch(show_api=False)