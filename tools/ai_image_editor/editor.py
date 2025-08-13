# deepface/tools/ai_image_editor/editor.py

"""
AI Image Editor with Face Verification, Face Analysis, and optional Chatbot.
Integrates with DeepFace's analysis and verification methods.
"""

import os
import gradio as gr
import numpy as np
from PIL import Image
from deepface import DeepFace

from . import detection
from . import chatbot


# ---------- Image Processing Functions ----------

def process_image(img, operation, threshold, width, height, scale_factor, angle, mask_diameter, object_type):
    """Run the selected image processing operation."""
    if not img:
        return None
    ops = {
        "Detect Faces": lambda: detection.detect(img, threshold),
        "Resize": lambda: detection.resize(img, width, height),
        "Rescale": lambda: detection.rescale(img, scale_factor),
        "Rotate": lambda: detection.rotate(img, angle),
        "Masked Image": lambda: detection.masked_image(img, mask_diameter),
        "Smart Blur": lambda: detection.smart_blur(img),
        "Remove Objects": lambda: detection.object_remover(img, object_type) if object_type else img
    }
    return ops.get(operation, lambda: img)()


def verify_faces(img1, img2):
    """Verify two uploaded images."""
    if not img1 or not img2:
        return None, "Please upload both images."
    result = detection.face_verify(img1, img2)
    return None, f"✅ SAME PERSON" if result else "❌ DIFFERENT PERSON"


def face_features(img):
    """Analyze face features in an image."""
    if not img:
        return None, "Please upload an image."
    try:
        img_array = np.array(img)
        if img_array.shape[-1] == 4:
            img_array = img_array[:, :, :3]
        result = DeepFace.analyze(
            img_array,
            actions=['age', 'gender', 'emotion', 'race'],
            enforce_detection=False,
            detector_backend='opencv'
        )
        if isinstance(result, list):
            result = result[0]
        status = f"Age: {result.get('age', 'N/A')} | " \
                 f"Gender: {result.get('gender', 'N/A')} | " \
                 f"Emotion: {result.get('dominant_emotion', 'N/A')} | " \
                 f"Race: {result.get('dominant_race', 'N/A')}"
        return img, status
    except Exception as e:
        return None, f"Analysis failed: {e}"


# ---------- Gradio Interface ----------

def launch_editor():
    """Launch the Gradio interface for the AI Image Editor."""
    with gr.Blocks(title="AI Image Editor") as demo:
        gr.Markdown("# AI Image Editor with Face Verification")

        with gr.Tabs():
            # Chatbot Tab (only if enabled)
            if chatbot.is_enabled():
                with gr.TabItem("AI Assistant"):
                    gr.Markdown("### Ask the assistant to guide you to the right feature.")
                    chat_box = gr.Chatbot(label="Chat", height=400)
                    msg = gr.Textbox(label="Your message", placeholder="Type here...")
                    clear = gr.Button("Clear Chat")

                    def respond(message, chat_history):
                        if not message:
                            return chat_history, ""
                        conv_history = []
                        for u, a in chat_history:
                            conv_history.append({'role': 'user', 'content': u})
                            conv_history.append({'role': 'assistant', 'content': a})
                        bot_message = chatbot.get_response(message, conv_history)
                        chat_history.append((message, bot_message))
                        return chat_history, ""

                    msg.submit(respond, [msg, chat_box], [chat_box, msg])
                    clear.click(lambda: ([], ""), outputs=[chat_box, msg])
            else:
                with gr.TabItem("AI Assistant"):
                    gr.Markdown("⚠️ Chatbot disabled. Set your OPENAI_API_KEY environment variable to enable.")

            # Image Processing Tab
            with gr.TabItem("Image Processing"):
                with gr.Row():
                    with gr.Column():
                        img_input = gr.Image(type="pil", label="Input Image", sources=["upload", "webcam"])
                        operation = gr.Dropdown(
                            ["Detect Faces", "Resize", "Rescale", "Rotate", "Masked Image", "Smart Blur", "Remove Objects"],
                            value="Detect Faces", label="Operation"
                        )
                        threshold = gr.Slider(0, 1, value=0.5, label="Confidence Threshold", visible=True)
                        width = gr.Number(value=256, label="Width", visible=False)
                        height = gr.Number(value=256, label="Height", visible=False)
                        scale_factor = gr.Slider(0.1, 3.0, value=1.0, step=0.01, label="Scale Factor", visible=False)
                        angle = gr.Slider(0, 360, value=0, step=1, label="Angle", visible=False)
                        mask_diameter = gr.Slider(1, 512, value=128, step=1, label="Circle Diameter", visible=False)
                        object_type = gr.Dropdown(choices=[], label="Object Type to Remove", visible=False)
                        btn = gr.Button("Apply")
                    with gr.Column():
                        img_output = gr.Image(type="pil", label="Output Image")

                def update_visibility(op):
                    return {
                        threshold: gr.update(visible=(op == "Detect Faces")),
                        width: gr.update(visible=(op == "Resize")),
                        height: gr.update(visible=(op == "Resize")),
                        scale_factor: gr.update(visible=(op == "Rescale")),
                        angle: gr.update(visible=(op == "Rotate")),
                        mask_diameter: gr.update(visible=(op == "Masked Image")),
                        object_type: gr.update(visible=(op == "Remove Objects")),
                    }

                operation.change(update_visibility, operation, [threshold, width, height, scale_factor, angle, mask_diameter, object_type])
                btn.click(process_image, [img_input, operation, threshold, width, height, scale_factor, angle, mask_diameter, object_type], img_output)

            # Face Verification Tab
            with gr.TabItem("Face Verification"):
                gr.Markdown("### Upload two images to compare faces")
                with gr.Row():
                    with gr.Column():
                        img1_upload = gr.Image(type="pil", label="Image 1", sources=["upload"])
                        img2_upload = gr.Image(type="pil", label="Image 2", sources=["upload"])
                        verify_btn = gr.Button("Verify Faces")
                    with gr.Column():
                        verification_result = gr.Image(type="pil", label="Result Image (optional)")
                        status_text = gr.Textbox(label="Status", interactive=False, value="Ready to verify faces...")
                verify_btn.click(verify_faces, [img1_upload, img2_upload], [verification_result, status_text])

            # Face Analysis Tab
            with gr.TabItem("Face Analysis"):
                gr.Markdown("### Upload an image to analyze face features")
                with gr.Row():
                    with gr.Column():
                        analysis_img = gr.Image(type="pil", label="Image to Analyze", sources=["upload"])
                        analyze_btn = gr.Button("Analyze Face")
                    with gr.Column():
                        analysis_result = gr.Image(type="pil", label="Image (unchanged)")
                        analysis_status = gr.Textbox(label="Results", interactive=False, value="Ready to analyze...")
                analyze_btn.click(face_features, [analysis_img], [analysis_result, analysis_status])

    demo.launch(server_name="0.0.0.0", server_port=8080, share=False)


if __name__ == "__main__":
    launch_editor()
