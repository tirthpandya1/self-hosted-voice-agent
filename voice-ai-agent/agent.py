# agent.py
import os
import subprocess
import ollama
import chromadb
from sentence_transformers import SentenceTransformer
import sounddevice as sd
import numpy as np
import wave
import simpleaudio as sa # For playback
import json
import torch

# Required for Indic Parler TTS model and its tokenizers
from parler_tts import ParlerTTSForConditionalGeneration 
from transformers import AutoTokenizer

# --- Configuration ---
# STT (Whisper.cpp)
# Ensure this path is correct for your Windows setup.
# Using raw string or double backslashes for Windows paths is safer.
WHISPER_CPP_DIR = r"C:\whisper.cpp" # Example: r"C:\Users\YourUser\path\to\whisper.cpp"
WHISPER_MODEL_PATH = os.path.join(WHISPER_CPP_DIR,"models/ggml-large-v3-turbo.bin") # Ensure this model is in your 'models' folder
# Common path for Release build of whisper.cpp's main executable on Windows
# Verify this path matches your actual compiled executable location
WHISPER_EXECUTABLE = os.path.join(WHISPER_CPP_DIR, "build", "bin", "Release","whisper-cli.exe") 
RECORDING_FILE = "audio_files/temp_recording.wav"

# TTS (Indic Parler TTS)
PARLER_MODEL_NAME = "ai4bharat/indic-parler-tts"
DESCRIPTION_HI = "Divya speaks in a clear, moderate-paced voice. The recording is of very high quality with no background noise."
DESCRIPTION_EN = "Mary speaks in a clear, moderate-paced voice with an Indian English accent. The recording is of very high quality with no background noise."
DESCRIPTION_DEFAULT = "A speaker delivers speech in a clear, moderate-paced voice. The recording is of very high quality with no background noise."

# LLM (Ollama)
#OLLAMA_MODEL = "llama3:8b" 
OLLAMA_MODEL = "llama3.2:3b" 

# RAG
CHROMA_DB_PATH = "db/chroma_db"
COLLECTION_NAME = "voice_agent_docs"
EMBEDDING_MODEL_NAME = 'intfloat/multilingual-e5-large' # Changed back as per your file
RAG_TOP_K = 3

# Audio Recording
SAMPLE_RATE = 16000 
CHANNELS = 1
DURATION = 7 

# --- Initialize Components ---
print("Initializing components...")
tts_device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Using device for TTS: {tts_device}")

# Embedding Model
print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

# Vector DB
print(f"Connecting to Vector DB at: {CHROMA_DB_PATH}")
chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
try:
    rag_collection = chroma_client.get_collection(name=COLLECTION_NAME)
    print(f"Successfully connected to RAG collection: {COLLECTION_NAME}")
except Exception as e:
    print(f"Error connecting to Chroma collection: {e}. Make sure you've run ingest_docs.py.")
    exit()

# Indic Parler TTS Model and Tokenizers
parler_model = None
parler_prompt_tokenizer = None
parler_description_tokenizer = None
try:
    print(f"Loading Indic Parler TTS model ({PARLER_MODEL_NAME}) and tokenizers... This may take a while.")
    parler_model = ParlerTTSForConditionalGeneration.from_pretrained(PARLER_MODEL_NAME).to(tts_device)
    # Use AutoTokenizer directly from transformers library
    parler_prompt_tokenizer = AutoTokenizer.from_pretrained(PARLER_MODEL_NAME) # <<< CORRECTED
    parler_description_tokenizer = AutoTokenizer.from_pretrained(parler_model.config.text_encoder._name_or_path) # <<< CORRECTED
    print("Indic Parler TTS loaded successfully.")
except Exception as e:
    print(f"Fatal Error loading Indic Parler TTS: {e}")
    print("Please ensure you have a stable internet connection for the first download, sufficient disk space, and compatible PyTorch/CUDA versions.")
    import traceback
    traceback.print_exc() # Print full traceback for debugging
    parler_model = None 

print("All components initialized (or attempted).")

# --- Helper Functions ---
def record_audio(filename=RECORDING_FILE, duration=DURATION, rate=SAMPLE_RATE, channels=CHANNELS):
    print(f"\nRecording for {duration} seconds... Speak now!")
    recording = sd.rec(int(duration * rate), samplerate=rate, channels=channels, dtype='int16')
    sd.wait() 
    
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2) 
        wf.setframerate(rate)
        wf.writeframes(recording.tobytes())
    print(f"Recording saved to {filename}")
    return filename

def transcribe_audio_whisper_cpp(audio_path):
    print("Transcribing audio with Whisper.cpp...")
    
    abs_whisper_executable = os.path.abspath(WHISPER_EXECUTABLE)
    abs_whisper_model_path = os.path.abspath(WHISPER_MODEL_PATH)
    abs_audio_path = os.path.abspath(audio_path)

    if not os.path.exists(abs_whisper_executable):
        print(f"Whisper executable not found at: {abs_whisper_executable}")
        return None, None
    if not os.path.exists(abs_whisper_model_path):
        print(f"Whisper model not found at: {abs_whisper_model_path}")
        return None, None

    # Define the expected JSON output filename
    expected_json_filename = f"{abs_audio_path}.json"
    # Remove pre-existing JSON file to ensure we read the new one
    if os.path.exists(expected_json_filename):
        try:
            os.remove(expected_json_filename)
        except Exception as e:
            print(f"Warning: Could not remove pre-existing JSON file {expected_json_filename}: {e}")


    command = [
        abs_whisper_executable,
        "-m", abs_whisper_model_path,
        "-f", abs_audio_path,
        "-l", "auto", 
        "-oj" # Output JSON to file: <audio_path>.json
    ]
    print(f"Executing Whisper.cpp command: {' '.join(command)}")
    
    try:
        process = subprocess.run(command, capture_output=True, text=True, check=True, 
                                 cwd=os.path.dirname(abs_whisper_executable), encoding='utf-8')
        
        # Log stdout (which contains plain text) and stderr for debugging
        print("Whisper.cpp STDOUT:\n", process.stdout)
        print("Whisper.cpp STDERR:\n", process.stderr)

        # Check if the JSON file was created
        if not os.path.exists(expected_json_filename):
            print(f"Error: Whisper.cpp did not create the expected JSON output file: {expected_json_filename}")
            print("Please check Whisper.cpp execution and permissions.")
            return None, None

        # Read the JSON content from the file
        with open(expected_json_filename, 'r', encoding='utf-8') as f:
            transcription_json = json.load(f)
        
        # Clean up the created JSON file after reading
        try:
            os.remove(expected_json_filename)
        except Exception as e:
            print(f"Warning: Could not remove JSON file {expected_json_filename} after processing: {e}")

        full_text = ""
        # The JSON structure from whisper.cpp when output to file is usually an object
        # with a "transcription" key which is an array of segments.
        # Or sometimes a simpler "text" key directly if -otxt is also used (but we use -oj)
        # Let's adapt to the common structure for -oj file output
        segments = transcription_json.get("transcription", [])
        if isinstance(segments, list):
            full_text = " ".join([segment.get("text", "").strip() for segment in segments]).strip()
        elif "text" in transcription_json: # Fallback for simpler structure
            full_text = transcription_json["text"].strip()
        else: # If transcription key is not found, try to iterate over segments directly if JSON is a list
            if isinstance(transcription_json, list): # Some versions might output a list of segments directly
                 full_text = " ".join([segment.get("text", "").strip() for segment in transcription_json]).strip()


        detected_language = "unknown"
        # Attempt to get overall language detection if present
        if "language" in transcription_json:
            lang_info = transcription_json["language"]
            if isinstance(lang_info, str): # Simplest case: "language": "en"
                detected_language = lang_info
            elif isinstance(lang_info, dict): # Common case: "language": {"language": "en", "probability": 0.9, ...}
                detected_language = lang_info.get("language", lang_info.get("lang", "unknown")) # Check for 'language' or 'lang' key
        elif "params" in transcription_json and isinstance(transcription_json["params"], dict) and "language" in transcription_json["params"]:
            # Older whisper.cpp versions might have it in params
            detected_language = transcription_json["params"]["language"]
        
        # If still unknown and segments exist, try to infer from the first segment's language property (if available)
        # Note: whisper.cpp JSON file output (-oj) might not always have per-segment language.
        # The overall detection is usually more reliable if present.
        if detected_language == "unknown" and segments and isinstance(segments, list) and len(segments) > 0:
            first_segment = segments[0]
            if isinstance(first_segment, dict) and "language" in first_segment:
                 # This structure is less common in the file output, but good to have a check
                segment_lang_info = first_segment["language"]
                if isinstance(segment_lang_info, str):
                    detected_language = segment_lang_info
                elif isinstance(segment_lang_info, dict):
                    detected_language = segment_lang_info.get("language", segment_lang_info.get("lang", "unknown"))
            # Fallback: Check if whisper's internal detection was logged to stderr and parse it (more complex)
            # For now, we rely on the JSON structure. The whisper.cpp log shows "auto-detected language: hi" in stderr.

        print(f"Transcription from JSON: {full_text}")
        print(f"Detected language from JSON (by Whisper): {detected_language}") # Should be 'hi' now
        return full_text, detected_language.split('-')[0].lower() # Return only primary lang code e.g. 'hi'

    except subprocess.CalledProcessError as e:
        print(f"Error during transcription with Whisper.cpp: {e}")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        return None, None
    except FileNotFoundError as e: # For the JSON file
        print(f"Error: JSON output file not found: {e}")
        return None, None
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from file {expected_json_filename}: {e}")
        # Optionally, print the problematic file content if small enough
        return None, None
    except Exception as e:
        print(f"An unexpected error occurred during transcription: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def get_rag_context(query_text, top_k=RAG_TOP_K):
    if not query_text:
        return ""
    print(f"Retrieving RAG context for query: {query_text}")
    query_embedding = embedding_model.encode(f"query: {query_text}").tolist()
    results = rag_collection.query(query_embeddings=[query_embedding], n_results=top_k)
    context = "\n".join(results['documents'][0]) if results['documents'] and results['documents'][0] else ""
    print(f"Retrieved context:\n---\n{context if context else 'No context found.'}\n---")
    return context

def query_llm(user_query, rag_context, input_lang="en"):
    print("Querying LLM...")
    system_prompt = (
        "You are a helpful multilingual AI assistant. "
        "The user might speak in English, Hindi, or Hinglish (a mix of Hindi and English). "
        "Understand the user's query and use the provided context to answer accurately. "
        "If the context is not relevant, answer based on your general knowledge but state that the context was not helpful. "
        "Be concise. "
    )
    if input_lang == "hi": 
         system_prompt += "Please try to respond primarily in clear Hindi as the user spoke Hindi. If you must use English words, keep them minimal."
    else: 
         system_prompt += "Please respond primarily in clear English."

    prompt = f"""Context from documents:
---
{rag_context if rag_context else "No relevant context found in local documents."}
---
User Query: {user_query}

Assistant Response:"""
    try:
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
        )
        llm_response_text = response['message']['content']
        print(f"LLM Response: {llm_response_text}")
        return llm_response_text
    except Exception as e:
        print(f"Error querying Ollama: {e}")
        return "Sorry, I encountered an error while processing your request."

def synthesize_speech_indic_parler(text_to_speak, lang_code="en"):
    if not parler_model or not parler_prompt_tokenizer or not parler_description_tokenizer:
        print("Indic Parler TTS model/tokenizers not loaded. Cannot synthesize.")
        return None, 24000  

    print(f"Synthesizing with Indic Parler TTS for lang_code: {lang_code}, text: '{text_to_speak[:60]}...'")

    if lang_code == "hi":
        description = DESCRIPTION_HI
    elif lang_code == "en":
        description = DESCRIPTION_EN
    else: 
        description = DESCRIPTION_DEFAULT 
    try:
        # Tokenize prompt (text to be spoken)
        prompt_inputs = parler_prompt_tokenizer(text_to_speak, return_tensors="pt", padding=True, truncation=True, max_length=512) # Added padding, truncation
        prompt_input_ids = prompt_inputs.input_ids.to(tts_device)
        prompt_attention_mask = prompt_inputs.attention_mask.to(tts_device)
        
        # Tokenize description (style prompt)
        description_inputs = parler_description_tokenizer(description, return_tensors="pt", padding=True, truncation=True, max_length=128) # Added padding, truncation
        description_input_ids = description_inputs.input_ids.to(tts_device)
        description_attention_mask = description_inputs.attention_mask.to(tts_device)

        generation = parler_model.generate(
            input_ids=description_input_ids, # This is the style/speaker description
            attention_mask=description_attention_mask,
            prompt_input_ids=prompt_input_ids, # This is the text to be synthesized
            prompt_attention_mask=prompt_attention_mask,
            do_sample=True, 
            temperature=0.6, 
            top_k=50,
            top_p=0.95,
        ).cpu() # Move to CPU after generation for numpy conversion
        
        audio_arr = generation.numpy().squeeze()
        
        if audio_arr.ndim > 1:
            audio_arr = audio_arr.flatten()

        if audio_arr.dtype == np.float32:
            audio_arr_int16 = (audio_arr * 32767).astype(np.int16)
        elif audio_arr.dtype == np.int16:
            audio_arr_int16 = audio_arr
        else:
            print(f"Warning: Unexpected audio array dtype: {audio_arr.dtype}. Trying to convert.")
            audio_arr_int16 = audio_arr.astype(np.int16)
            
        audio_bytes = audio_arr_int16.tobytes()
        tts_sample_rate = parler_model.config.sampling_rate
        
        print(f"Indic Parler TTS synthesis successful. Sample rate: {tts_sample_rate}")
        return audio_bytes, tts_sample_rate
        
    except Exception as e:
        print(f"Error during Indic Parler TTS synthesis: {e}")
        import traceback
        traceback.print_exc()
        return None, parler_model.config.sampling_rate if parler_model else 24000

def play_audio_bytes(audio_bytes, sample_rate, sample_width=2, channels=CHANNELS):
    if not audio_bytes:
        print("No audio data to play.")
        return
    
    print(f"Playing audio: {len(audio_bytes)} bytes, Sample rate: {sample_rate} Hz, Width: {sample_width}, Channels: {channels}")
    
    # Basic sanity check: length should be multiple of (sample_width * channels)
    frame_size = sample_width * channels
    if len(audio_bytes) % frame_size != 0:
        print(f"Warning: Audio data length ({len(audio_bytes)}) is not a multiple of frame size ({frame_size}). This might indicate corruption.")

    try:
        play_obj = sa.play_buffer(audio_bytes, num_channels=channels, bytes_per_sample=sample_width, sample_rate=sample_rate)
        play_obj.wait_done()
        print("Playback finished.")
    except Exception as e:
        print(f"Error during simpleaudio playback: {e}")
        import traceback
        traceback.print_exc()


# --- Main Loop ---
if __name__ == "__main__":
    if not os.path.exists(WHISPER_EXECUTABLE):
        print(f"Whisper.cpp executable not found at '{os.path.abspath(WHISPER_EXECUTABLE)}'")
        print("Please ensure whisper.cpp is compiled and WHISPER_EXECUTABLE path is correct (check for .exe on Windows).")
        exit(1)
    if not parler_model:
        print("Indic Parler TTS model failed to load. Exiting.")
        exit(1)

    try:
        while True:
            action = input("\nPress Enter to record, or type 'q' to quit: ")
            if action.lower() == 'q':
                break

            audio_file = record_audio()
            user_query_text, detected_lang_code_whisper = transcribe_audio_whisper_cpp(audio_file)

            if not user_query_text:
                print("Could not transcribe audio. Please try again.")
                err_audio_bytes, err_sr = synthesize_speech_indic_parler("Sorry, I could not understand that.", lang_code="en")
                if err_audio_bytes: play_audio_bytes(err_audio_bytes, err_sr)
                continue
            
            primary_lang_code_for_llm = detected_lang_code_whisper.split('-')[0].lower() if detected_lang_code_whisper else "en"

            rag_context = get_rag_context(user_query_text)
            llm_response = query_llm(user_query_text, rag_context, input_lang=primary_lang_code_for_llm)

            tts_speaker_lang_preference = "en" 
            if primary_lang_code_for_llm == "hi":
                tts_speaker_lang_preference = "hi"
            
            audio_output_bytes, tts_sr = synthesize_speech_indic_parler(llm_response, lang_code=tts_speaker_lang_preference)
            
            if audio_output_bytes:
                play_audio_bytes(audio_output_bytes, sample_rate=tts_sr)
            else:
                print("Could not synthesize speech for the response.")
                fb_audio, fb_sr = synthesize_speech_indic_parler("I have a response, but encountered an issue speaking it.", lang_code="en")
                if fb_audio: play_audio_bytes(fb_audio, fb_sr)

    except KeyboardInterrupt:
        print("\nExiting agent...")
    except Exception as e:
        print(f"An unexpected error occurred in the main loop: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if os.path.exists(RECORDING_FILE):
            try:
                os.remove(RECORDING_FILE)
            except Exception as e: # More general exception
                print(f"Could not remove temp file {RECORDING_FILE}: {e}")
        print("Cleanup complete.")