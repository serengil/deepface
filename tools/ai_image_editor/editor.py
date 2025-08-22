# deepface/tools/ai_image_editor/editor.py

"""
AI Image Editor with Face Verification, Face Analysis, and optional Chatbot.
Integrates with DeepFace's analysis and verification methods.
"""

import os
import gradio as gr
import tensorflow as tf
import cv2
import numpy as np
from PIL import Image
from deepface import DeepFace
import threading
import subprocess
import sys


from detection import detect, resize, rescale, rotate, masked_image, smart_blur, get_detected_object_types, object_remover, face_verify
from chatbot import response as chatbot_response

def process_image(img, operation, threshold, width, height, scale_factor, angle, mask_diameter, object_type):
    if operation == "Detect Faces":
        return detect(img, threshold)
    elif operation == "Resize":
        return resize(img, width, height)
    elif operation == "Rescale":
        return rescale(img, scale_factor)
    elif operation == "Rotate":
        return rotate(img, angle)
    elif operation == "Masked Image":
        return masked_image(img, mask_diameter)
    elif operation == "Smart Blur":
        return smart_blur(img)
    elif operation == "Remove Objects":
        if object_type:
            result = object_remover(img, object_type)
            # object_remover returns (image, objects_list), we only need the image
            if isinstance(result, tuple):
                return result[0]
            return result
        return img
    return img

def verify_faces(img1, img2, label1="Image 1", label2="Image 2"):
    if not img1 or not img2:
        return None, "Hi user, Please Upload both images first!"
    
    try:
        img1_array = np.array(img1)
        img2_array = np.array(img2)

        if img1_array.shape[-1] == 4:
            img1_array = img1_array[:, :, :3]
        if img2_array.shape[-1] == 4:
            img2_array = img2_array[:, :, :3]
        
        # Use VGG-Face model with cosine distance for better accuracy
        MODEL_NAME = "VGG-Face"
        THRESHOLD = 0.8  # Optimal threshold based on test results
            
        IMAGE_SIZE = 224  # VGG-Face standard size
        try:
            img1_resized = cv2.resize(img1_array, (IMAGE_SIZE, IMAGE_SIZE))
            img2_resized = cv2.resize(img2_array, (IMAGE_SIZE, IMAGE_SIZE))
            
            result = DeepFace.verify(
                img1_resized, 
                img2_resized, 
                model_name=MODEL_NAME, 
                enforce_detection=False,
                distance_metric="cosine",  # Best for VGG-Face
                threshold=THRESHOLD
            )
        except Exception as e:
            # Fallback to correlation method if DeepFace fails
            try:
                img1_gray = cv2.cvtColor(img1_resized, cv2.COLOR_RGB2GRAY)
                img2_gray = cv2.cvtColor(img2_resized, cv2.COLOR_RGB2GRAY)
                correlation = np.corrcoef(img1_gray.flatten(), img2_gray.flatten())[0, 1]
                if np.isnan(correlation):
                    correlation = 0
                if correlation > 0.5:  # Reasonable correlation threshold for same person
                    result = {'verified': True, 'distance': 1 - correlation}
                else:
                    result = {'verified': False, 'distance': 1 - correlation}
            except Exception as e:
                return None, f"Model loading failed. Please try again. Error: {str(e)}"
        
        img2_resized = cv2.resize(img2_array, (img1_array.shape[1], img1_array.shape[0]))
        combined = np.hstack([img1_array, img2_resized])
        
        confidence = 1 - result['distance']
        if result['verified']:
            text = f"✅ SAME PERSON (Confidence: {confidence:.2f})"
            color = (0, 255, 0)
            status = f"✅ SAME PERSON (Confidence: {confidence:.2f})"
        else:
            text = f"❌ DIFFERENT PERSON (Confidence: {confidence:.2f})"
            color = (0, 0, 255)
            status = f"❌ DIFFERENT PERSON (Confidence: {confidence:.2f})"
        
        cv2.putText(combined, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(combined, label1, (10, combined.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(combined, label2, (img1_array.shape[1] + 10, combined.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return combined, status
        
    except Exception as e:
        return None, f"Error: {str(e)}"

def verify_webcam_faces(img1, img2, label1="Reference", label2="Webcam"):
    if not img1 or not img2:
        return None, "Hi user, Please Upload reference image and capture webcam photo!"
    
    try:
        img1_array = np.array(img1)
        img2_array = np.array(img2)

        if img1_array.shape[-1] == 4:
            img1_array = img1_array[:, :, :3]
        if img2_array.shape[-1] == 4:
            img2_array = img2_array[:, :, :3]
        
        # Use VGG-Face model with cosine distance for better accuracy
        MODEL_NAME = "VGG-Face"
        THRESHOLD = 0.4  # Standard threshold for VGG-Face with cosine
            
        IMAGE_SIZE = 224  # VGG-Face standard size
        try:
            img1_resized = cv2.resize(img1_array, (IMAGE_SIZE, IMAGE_SIZE))
            img2_resized = cv2.resize(img2_array, (IMAGE_SIZE, IMAGE_SIZE))
            
            result = DeepFace.verify(
                img1_resized, 
                img2_resized, 
                model_name=MODEL_NAME, 
                enforce_detection=False,
                distance_metric="cosine",  # Best for VGG-Face
                threshold=THRESHOLD
            )
        except Exception as e:
            # Fallback to correlation method if DeepFace fails
            try:
                img1_gray = cv2.cvtColor(img1_resized, cv2.COLOR_RGB2GRAY)
                img2_gray = cv2.cvtColor(img2_resized, cv2.COLOR_RGB2GRAY)
                correlation = np.corrcoef(img1_gray.flatten(), img2_gray.flatten())[0, 1]
                if np.isnan(correlation):
                    correlation = 0
                if correlation > 0.7:  # High correlation threshold for webcam same person
                    result = {'verified': True, 'distance': 1 - correlation}
                else:
                    result = {'verified': False, 'distance': 1 - correlation}
            except Exception as e:
                return None, f"Model loading failed. Please try again. Error: {str(e)}"
        
        img2_resized = cv2.resize(img2_array, (img1_array.shape[1], img1_array.shape[0]))
        combined = np.hstack([img1_array, img2_resized])
        
        confidence = 1 - result['distance']
        if result['verified']:
            text = f"✅ SAME PERSON (Confidence: {confidence:.2f})"
            color = (0, 255, 0)
            status = f"✅ SAME PERSON (Confidence: {confidence:.2f})"
        else:
            text = f"❌ DIFFERENT PERSON (Confidence: {confidence:.2f})"
            color = (0, 0, 255)
            status = f"❌ DIFFERENT PERSON (Confidence: {confidence:.2f})"
        
        cv2.putText(combined, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(combined, label1, (10, combined.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(combined, label2, (img1_array.shape[1] + 10, combined.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return combined, status
        
    except Exception as e:
        return None, f"Error: {str(e)}"

def face_features(img):
    if not img:
        return None, "Please upload an image first!"
    
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
        
        # Fix the output format to display clean text
        age = result.get('age', 'N/A')
        
        # Handle gender output properly
        gender_data = result.get('gender', {})
        if isinstance(gender_data, dict):
            # Find the gender with highest probability
            if gender_data:
                dominant_gender = max(gender_data.items(), key=lambda x: float(x[1]))[0]
                gender = dominant_gender
            else:
                gender = 'N/A'
        else:
            gender = str(gender_data)
        
        # Handle emotion output properly
        emotion_data = result.get('dominant_emotion', 'N/A')
        if isinstance(emotion_data, dict):
            if emotion_data:
                dominant_emotion = max(emotion_data.items(), key=lambda x: float(x[1]))[0]
                emotion = dominant_emotion
            else:
                emotion = 'N/A'
        else:
            emotion = str(emotion_data)
        
        # Handle race output properly
        race_data = result.get('dominant_race', 'N/A')
        if isinstance(race_data, dict):
            if race_data:
                dominant_race = max(race_data.items(), key=lambda x: float(x[1]))[0]
                race = dominant_race
            else:
                race = 'N/A'
        else:
            race = str(race_data)
        
        status = f"Age: {age} | Gender: {gender} | Emotion: {emotion} | Race: {race}"
        
        return img, status
        
    except Exception as e:
        return None, f"Analysis failed. Try a different image. Error: {str(e)}"



def get_object_types(img, operation):
    if img and operation == "Remove Objects":
        types = get_detected_object_types(img)
        return gr.update(choices=types, visible=True, value=types[0] if types else None)
    return gr.update(choices=[], visible=False, value=None)

def enable_webcam():
    return gr.update(sources=["upload", "webcam"])

def run_chatbot():
    try:
        subprocess.run([sys.executable, "chatbot.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Chatbot process error: {e}")
    except Exception as e:
        print(f"Error running chatbot: {e}")



with gr.Blocks(title="AI Image Editor") as demo:
    gr.Markdown("# AI Image Editor with Face Verification")
    
    with gr.Tabs():
        with gr.TabItem("AI Assistant"):
            gr.Markdown("### Welcome! I'm your AI assistant. Tell me what you want to do and I'll guide you to the right feature.")
            
            chatbot = gr.Chatbot(label="Chat with AI Assistant", height=400)
            msg = gr.Textbox(label="Your message", placeholder="Tell me what you want to do...")
            clear = gr.Button("Clear Chat")
            
            def respond(message, chat_history):
                if not message:
                    return chat_history, ""
                
                conversation_history = []
                for user_msg, bot_msg in chat_history:
                    conversation_history.append({'role': 'user', 'content': user_msg})
                    conversation_history.append({'role': 'assistant', 'content': bot_msg})
                
                bot_message = chatbot_response(message, conversation_history)
                chat_history.append((message, bot_message))
                return chat_history, ""
            
            msg.submit(respond, [msg, chatbot], [chatbot, msg])
            clear.click(lambda: ([], ""), outputs=[chatbot, msg])
        
        with gr.TabItem("Image Processing"):
            with gr.Row():
                with gr.Column():
                    img_input = gr.Image(type="pil", label="Input Image", sources=["upload", "webcam"])
                    operation = gr.Dropdown([
                        "Detect Faces", "Resize", "Rescale", "Rotate", 
                        "Masked Image", "Smart Blur", "Remove Objects"
                    ], value="Detect Faces", label="Operation")
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

        
        with gr.TabItem("Face Verification"):
            with gr.TabItem("Photo Upload"):
                gr.Markdown("### Upload two images to compare faces")
                gr.Markdown("**Best for:** High quality photos")
                
                with gr.Row():
                    with gr.Column():
                        img1_upload = gr.Image(type="pil", label="Image 1", sources=["upload"])
                        img2_upload = gr.Image(type="pil", label="Image 2", sources=["upload"])
                        verify_upload_btn = gr.Button("Verify Faces", variant="primary", size="lg")
                    with gr.Column():
                        verification_upload_result = gr.Image(type="pil", label="Verification Result")
                        status_upload_text = gr.Textbox(label="Status", interactive=False, value="Ready to verify faces...")
            
            with gr.TabItem("Webcam Verification"):
                gr.Markdown("### Step 1: Upload reference image")
                gr.Markdown("### Step 2: Click the camera icon to capture webcam photo")
                gr.Markdown("### Step 3: Click 'Verify Faces'")
                gr.Markdown("**Best for:** Lower quality webcam images")
                
                with gr.Row():
                    with gr.Column():
                        reference_img = gr.Image(type="pil", label="Reference Image", sources=["upload"])
                        webcam_btn = gr.Button("📷 Use Webcam", variant="secondary", size="sm")
                        webcam_img = gr.Image(type="pil", label="Webcam Photo", sources=["upload"], streaming=False)
                        verify_btn = gr.Button("Verify Faces", variant="primary", size="lg")
                    with gr.Column():
                        verification_result = gr.Image(type="pil", label="Verification Result")
                        status_text = gr.Textbox(label="Status", interactive=False, value="Ready to verify faces...")
        
        with gr.TabItem("Face Analysis"):
            with gr.TabItem("Photo Analysis"):
                gr.Markdown("### Upload an image to analyze face features")
                gr.Markdown("**Analyzes:** Age, Gender, Emotion, Race")
                
                with gr.Row():
                    with gr.Column():
                        analysis_img = gr.Image(type="pil", label="Image to Analyze", sources=["upload"])
                        analyze_btn = gr.Button("Analyze Face", variant="primary", size="lg")
                    with gr.Column():
                        analysis_result = gr.Image(type="pil", label="Analysis Result")
                        analysis_status = gr.Textbox(label="Analysis Results", interactive=False, value="Ready to analyze...")
            
            with gr.TabItem("Webcam Analysis"):
                gr.Markdown("### Capture webcam photo for face analysis")
                gr.Markdown("**Analyzes:** Age, Gender, Emotion, Race")
                
                with gr.Row():
                    with gr.Column():
                        webcam_analysis_btn = gr.Button("📷 Use Webcam", variant="secondary", size="sm")
                        webcam_analysis_img = gr.Image(type="pil", label="Webcam Photo", sources=["upload"], streaming=False)
                        analyze_webcam_btn = gr.Button("Analyze Face", variant="primary", size="lg")
                    with gr.Column():
                        webcam_analysis_result = gr.Image(type="pil", label="Analysis Result")
                        webcam_analysis_status = gr.Textbox(label="Analysis Results", interactive=False, value="Ready to analyze...")
            

    
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
    img_input.change(get_object_types, [img_input, operation], object_type)
    operation.change(get_object_types, [img_input, operation], object_type)
    
    btn.click(process_image, [img_input, operation, threshold, width, height, scale_factor, angle, mask_diameter, object_type], img_output)
    webcam_btn.click(enable_webcam, outputs=webcam_img)
    verify_upload_btn.click(verify_faces, [img1_upload, img2_upload], [verification_upload_result, status_upload_text])
    verify_btn.click(verify_webcam_faces, [reference_img, webcam_img], [verification_result, status_text])
    analyze_btn.click(face_features, [analysis_img], [analysis_result, analysis_status])
    webcam_analysis_btn.click(enable_webcam, outputs=webcam_analysis_img)
    analyze_webcam_btn.click(face_features, [webcam_analysis_img], [webcam_analysis_result, webcam_analysis_status])

if __name__ == "__main__":
    
    chatbot_thread = threading.Thread(target=run_chatbot, daemon=True)
    chatbot_thread.start()
    
    demo.launch(server_name="0.0.0.0", server_port=8080, share=False)
