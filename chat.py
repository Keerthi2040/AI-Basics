import requests
import json
import gradio as gr

# API endpoint and headers
url = "http://localhost:11434/api/generate"
headers = {
    "Content-Type": "application/json",
}

# Initialize conversation history
conversation_history = []

def generate_response(prompt, model, temperature):
    # Append user prompt to conversation history
    conversation_history.append(f"User: {prompt}")

    # Join the conversation history to create the full prompt
    full_prompt = "\n".join(conversation_history)

    # Prepare the data for the API request
    data = {
        "model": model,
        "stream": False,
        "prompt": full_prompt,
        "options": {
            "temperature": temperature,
        }
    }

    # Send the request to the API
    response = requests.post(url, headers=headers, data=json.dumps(data))

    # Check if the request was successful
    if response.status_code == 200:
        response_text = response.text
        data = json.loads(response_text)
        actual_response = data["response"]
        
        # Append the model's response to the conversation history
        conversation_history.append(f"AI: {actual_response}")
        
        # Return the formatted conversation history
        return "\n".join(conversation_history)
    else:
        # Handle errors
        error_message = f"Error: {response.status_code}, {response.text}"
        print(error_message)
        return error_message

# Define the Gradio interface
iface = gr.Interface(
    fn=generate_response,
    inputs=[
        gr.Textbox(lines=2, placeholder="Enter your prompt here...", label="Your Message"),
        gr.Dropdown(choices=["llama3.2:1b", "llama3.2:7b", "llama3.2:13b"], label="Model", value="llama3.2:1b"),
        gr.Slider(minimum=0.1, maximum=1.0, step=0.1, value=0.7, label="Temperature")
    ],
    outputs=gr.Textbox(lines=10, label="Conversation History"),
    title="🤖 Chat with LLaMA 3.2",
    description="This is a chat interface powered by LLaMA 3.2. Choose a model and adjust the temperature to control the creativity of the responses.",
    theme="soft",
    live=False,
)

# Launch the interface
iface.launch(share=True)