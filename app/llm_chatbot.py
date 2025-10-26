from transformers import pipeline

# Lightweight local LLM
chatbot = pipeline("text-generation", model="distilgpt2")

def chatbot_response(prompt, df=None):
    context = ""
    if df is not None:
        counts = df['sentiment'].value_counts().to_dict()
        context = f"Current sentiment counts: {counts}\n"

    try:
        output = chatbot(context + prompt, max_new_tokens=100, do_sample=True, temperature=0.7)[0]['generated_text']
        return output[len(context + prompt):].strip()
    except Exception as e:
        return f"⚠️ Chatbot Error: {e}"
