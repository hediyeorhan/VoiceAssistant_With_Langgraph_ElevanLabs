import gradio as gr
from graph.graph import app
from tts_stt import record_and_transcribe, text_to_speech
from graph.history import ConversationHistory
import speech_recognition as sr
from elevenlabs.client import ElevenLabs
import os
from dotenv import load_dotenv

load_dotenv()

# Initialize components
client = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))
recognizer = sr.Recognizer()

# Conversation state
history = ConversationHistory(max_length=10)

def voice_assistant():
    custom_css = """
    .mic-container {
        display: flex;
        justify-content: center;
        align-items: center;
        height: 70vh;
    }
    .mic-button {
        width: 300px;
        height: 300px;
        border-radius: 50%;
        background: radial-gradient(circle, #6ec1e4, #0a3d62);
        box-shadow: 0 0 30px #ff4b4b;
        border: 8px solid white;
        font-size: 6em;
        color: white;
        display: flex;
        justify-content: center;
        align-items: center;
        cursor: pointer;
        transition: all 0.3s ease;
        animation: pulse 2s infinite;
        margin-top: 50px;
        margin-left: 450px;
    }
    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(255, 75, 75, 0.7); }
        70% { box-shadow: 0 0 0 25px rgba(255, 75, 75, 0); }
        100% { box-shadow: 0 0 0 0 rgba(255, 75, 75, 0); }
    }
    .mic-button:hover { transform: scale(1.05); }
    .mic-button:active { transform: scale(0.95); }
    .status-text { 
        text-align: center; 
        margin-top: 30px; 
        font-size: 1.5em;
        color: white;
    }
    body { background: linear-gradient(135deg, #1a1a2e, #16213e); }
    """

    with gr.Blocks(css=custom_css, title="Sesli Asistan") as demo:
        # Large circular microphone button
        mic_button = gr.Button("🎤", elem_classes="mic-button")
        
        # Status display
        status_display = gr.HTML("""
        <div class="status-text">
            Mikrofonu kullanmak için butona basın
        </div>
        """)
        
        # Hidden output
        response_output = gr.Textbox(visible=False)
        
        def process_voice():
            try:
                # Update status to listening
                yield {
                    status_display: """
                    <div style='text-align: center; color: #00ffff; font-weight: bold;'>
                        DİNLİYORUM... KONUŞUN
                    </div>
                    """
                }
                
                # Record and transcribe
                user_input = record_and_transcribe()
                if not user_input:
                    raise ValueError("Ses algılanamadı, lütfen tekrar deneyin")
                
                print(f"Algılanan: {user_input}")
                
                yield {
                    status_display: """
                    <div style='text-align: center; color: #00ffff; font-weight: bold;'>
                        DÜŞÜNÜYORUM...
                    </div>
                    """
                }
                
                # Process with AI
                config = {
                    "configurable": {
                        "thread_id": "default_thread",
                        "session_id": "default_session",
                        "checkpoint_ns": "default_ns",
                        "checkpoint_id": "default_id",
                        "language": "tr"
                    }
                }
                
                state = {
                    "question": user_input,
                    "history": history.get_history()
                }
                
                response = app.invoke(state, config=config)
                ai_response = response.get("generation", "Üzgünüm, bir yanıt oluşturamadım.")
                
                print(f"AI Yanıtı: {ai_response}")
                
                # Add to chat history
                history.add_interaction(user_input, ai_response)
                
                # Convert to speech
                text_to_speech(ai_response)
                
                # Update status with response
                yield {
                    response_output: ai_response,
                    status_display: f"""
                    <div style='text-align: center; #00ffff; font-weight: bold;'>
                        TAMAMLANDI: {ai_response[:100]}...
                    </div>
                    """
                }
                
            except Exception as e:
                error_msg = str(e)
                print(f"HATA: {error_msg}")
                yield {
                    status_display: f"""
                    <div style='text-align: center; color: #ff0000; font-weight: bold;'>
                        HATA: {error_msg}
                    </div>
                    """
                }

        mic_button.click(
            fn=process_voice,
            inputs=[],
            outputs=[response_output, status_display]
        )

    return demo

if __name__ == "__main__":
    demo = voice_assistant()
    demo.launch(share=False)