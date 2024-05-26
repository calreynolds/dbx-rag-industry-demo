import gradio as gr
from databricks.sdk import WorkspaceClient
from mlflow import deployments

# Volume link:
#  https://e2-dogfood.staging.cloud.databricks.com/explore/data/volumes/dbdemos/rag_chatbot_andrew_kraemer_newco/raw_document_landing_zone?o=6051921418418893

# Initialize the MLflow deployment client for Databricks
# Requires .databrickscfg file at /User/ directory if run locally
client = deployments.get_deploy_client("databricks")

# Initialize the Databricks WorkspaceClient
workspace_client = WorkspaceClient()

def respond_rag(message, chat_history):
    """
    Respond using the RAG model.
    
    Args:
    message (str): The user's message.
    chat_history (list): The chat history.

    Returns:
    tuple: The updated message and history.
    """
    if len(message.strip()) == 0:
        return "ERROR the question should not be empty"

    # Define the payload correctly as a dictionary with the 'inputs' key
    request_payload = {
        "inputs": [{"query": message}]
    }

    # Make a prediction using the RAG model
    response = client.predict(
        endpoint="dbdemos_rag_chatbot_andrew_kraemer_newco",
        inputs=request_payload
    )

    # Append the message and response to the history
    chat_history.append([message, response['predictions'][0]])

    return "", chat_history

def respond_base_dbrx(message, chat_history):
    """
    Respond using the base Databricks model.
    
    Args:
    message (str): The user's message.
    chat_history (list): The chat history.

    Returns:
    tuple: The updated message and chat history.
    """
    if len(message.strip()) == 0:
        return "ERROR the question should not be empty"

    # Define the messages payload
    messages = [
        {"role": "user", "content": message}
    ]

    # Make a prediction using the base Databricks model
    response = client.predict(
        endpoint="databricks-dbrx-instruct",
        inputs={"messages": messages, "max_tokens": 256}
    )

    # Append the message and response to the chat history
    chat_history.append((message, response.choices[0]["message"]["content"]))

    return "", chat_history

def download_and_upload_pdf(file, request : gr.Request):
    # Define local and DBFS paths

    # Pulls down the query parameter in the URL to place the PDF document in a specified place.
    folder_router = str(dict(request.query_params)["by"])

    volume_path = f"/Volumes/dbdemos/rag_chatbot_andrew_kraemer_newco/raw_document_landing_zone/pdf/{folder_router}/sample.pdf"

    # Upload the PDF to DBFS
    with open(file[0], "rb") as f:
        workspace_client.files.upload(volume_path, f.read(), overwrite=True) # type: ignore

    # Verify the file is uploaded
    resp = workspace_client.files.get_status(volume_path)
    print(resp)


def info_fn():
    gr.Info("Document submitted to Databricks Unity Catalog!")

# Define the Gradio theme
theme = gr.themes.Base(
    primary_hue="orange",
    secondary_hue="indigo",
    neutral_hue="slate",
    radius_size="sm",
).set(
    background_fill_primary='#FFFFFF',
    button_primary_background_fill="#1B3139",
    button_primary_background_fill_dark="#1B3139",
)

# Define custom CSS for styling
css = """
img {
    display: block;
    margin-left: auto;
    margin-right: auto;
}

h1 {
    text-align: center;
    display:block;
}
h2 {
    text-align: center;
    display:block;
}

# p, ul, ol {
#     margin-left: 40px; /* Adjust as needed */
#     margin-right: 40px; /* Adjust as needed */
# }
# li {
#     margin-left: 10px; /* Adjust the indentation of list items if necessary */
# }

"""

# Initialize the visibility flags for additional inputs
show_second_chat = None
show_pdf_input = None
show_pdf_submission_gif = None

# Build the Gradio interface
with gr.Blocks(theme=theme, css=css) as demo:
    # Header with logo and titles
    with gr.Row():
        gr.Image("./resources/dbx_logo.png", show_download_button=False, show_label=False, width="30%")
        with gr.Column(scale=3):
            gr.Markdown("# Databricks Apps")
            gr.Markdown("## Customizing DBRX with _your_  PDF document's context!")
            gr.Markdown("""This app lets you upload a PDF to customize the context for a DBRX model *on Databricks*. Start by uploading your PDF or use our preloaded ‘FiveTran, Best Practices Guide’ to ask questions like, ‘What is the best method of paging large amounts of data with FiveTran?’ The app uses both a RAG-enabled DBRX model and a standard DBRX model to provide answers. Use the checkboxes at the bottom of the screen to hide or show different features and view architecture diagrams.""")

    # PDF input section
    with gr.Column(visible=show_pdf_input) as pdf_textbox_col:
        with gr.Row():
            upload_button = gr.UploadButton("Upload PDF Document", file_types=[".pdf"], file_count="multiple")

    gr.HTML("<hr>")  # Horizontal line

    with gr.Column(visible=False) as pdf_submission_col:
        test_gif = gr.Gallery(["./resources/frog.gif"])
    with gr.Column(visible=False) as chatbot_arch_col:
        test_gif2 = gr.Gallery(["./resources/frog.gif"])        

    # Chatbot sections
    with gr.Row():
        with gr.Column():
            rag_chatbot = gr.Chatbot(label="RAG-enabled DBRX")

        with gr.Column(visible=show_second_chat) as second_chat_col:
            non_rag_chatbot = gr.Chatbot(label="Native DBRX")

    # User input and control buttons
    with gr.Row():
        user_msg = gr.Textbox(label="Enter your prompt", placeholder="Enter Prompt here!", min_width=1000)
        run_button = gr.Button(value="Send Message", size="sm", min_width=50)
        clear = gr.ClearButton([user_msg, rag_chatbot, non_rag_chatbot])

    gr.Button(value="Reset Demo", size="sm")

    # Toggle options for additional inputs
    with gr.Row():
        with gr.Column():
            show_second_chat = gr.Checkbox(label="Toggle non-RAG Chat", value=True)
            show_pdf_input = gr.Checkbox(label="Use custom PDF document", value=True)
        with gr.Column():
            show_architecture_one = gr.Checkbox(label="Toggle PDF Submission Architecture", value=False)
            show_architecture_two = gr.Checkbox(label="Toggle Chatbot Flow Architecture", value=False)

    # Define button click actions
    run_button.click(respond_rag, [user_msg, rag_chatbot], [user_msg, rag_chatbot])
    run_button.click(respond_base_dbrx, [user_msg, non_rag_chatbot], [user_msg, non_rag_chatbot])



    def toggle_gallery_visibility(current_visibility):
        return not current_visibility
    
    # Define visibility change actions

    # pdf_doc_submit_button.click(
    #     fn=pdf_doc_process,
    #     inputs=pdf_doc_submit_button,
    # )

    upload_value = upload_button.upload(download_and_upload_pdf, [upload_button], show_progress='full')
    upload_value = upload_button.upload(info_fn, [])

    print(upload_value.values)

    show_architecture_one.change(
        lambda show: {pdf_submission_col: gr.Column(visible=show)},
        inputs=show_architecture_one,
        outputs=pdf_submission_col
    )

    show_architecture_two.change(
        lambda show: {chatbot_arch_col: gr.Column(visible=show)},
        inputs=show_architecture_two,
        outputs=chatbot_arch_col
    )

    show_pdf_input.change(
        lambda show: {pdf_textbox_col: gr.Column(visible=show)},
        inputs=show_pdf_input,
        outputs=pdf_textbox_col
    )

    show_second_chat.change(
        lambda show: {second_chat_col: gr.Column(visible=show)},
        inputs=show_second_chat,
        outputs=second_chat_col
    )

# Launch the Gradio interface
demo.launch()
