import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Replace with your actual model name on Hugging Face Hub
model_name = "Yaminii/finetuned-mistral"

# Load the fine-tuned model and tokenizer from Hugging Face
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

# Define the recommendation function
def recommend_ingredients(user_input):
    input_ids = tokenizer(user_input, return_tensors="pt").input_ids.to(model.device)
    output = model.generate(input_ids, max_length=100)
    return tokenizer.decode(output[0], skip_special_tokens=True)

# Create the Gradio interface
iface = gr.Interface(
    fn=recommend_ingredients,
    inputs=gr.Textbox(placeholder="Describe your skin concerns..."),
    outputs=gr.Textbox(label="Recommended Ingredients"),
    title="AI-Powered Skincare Advisor",
    theme=gr.themes.Glass()  # Apply a glass theme
)

# Launch the app (Spaces automatically calls this)
iface.launch()
