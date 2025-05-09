# agent.py
import os
import subprocess
import ollama
import chromadb
from sentence_transformers import SentenceTransformer
import sounddevice as sd # Keep for recording, can use for playback test
import numpy as np
import wave
import simpleaudio as sa # For playback
import json
import torch
import soundfile as sf # For saving TTS output for debugging

from parler_tts import ParlerTTSForConditionalGeneration 
from transformers import AutoTokenizer

# --- Configuration ---
WHISPER_CPP_DIR = r"C:\whisper.cpp" 
WHISPER_MODEL_PATH = os.path.join(WHISPER_CPP_DIR, "models/ggml-large-v3-turbo.bin")
# VERIFY THIS EXECUTABLE NAME AND PATH
WHISPER_EXECUTABLE = os.path.join(WHISPER_CPP_DIR, "build", "bin", "Release", "whisper-cli.exe") 
RECORDING_FILE = "audio_files/temp_recording.wav"

PARLER_MODEL_NAME = "ai4bharat/indic-parler-tts"
DESCRIPTION_HI = "Divya speaks in a clear, moderate-paced voice. The recording is of very high quality with no background noise."
DESCRIPTION_EN = "Mary speaks in a clear, moderate-paced voice with an Indian English accent. The recording is of very high quality with no background noise."
DESCRIPTION_DEFAULT = DESCRIPTION_EN # Default to English speaker style if language is unknown

OLLAMA_MODEL = "llama3.2:3b" 
CHROMA_DB_PATH = "db/chroma_db"
COLLECTION_NAME = "voice_agent_docs"
EMBEDDING_MODEL_NAME = 'intfloat/multilingual-e5-large'
RAG_TOP_K = 3
SAMPLE_RATE = 16000 
CHANNELS = 1
DURATION = 7 

# --- Initialize Components ---
print("Initializing components...")
tts_device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Using device for TTS: {tts_device}")

embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
try:
    rag_collection = chroma_client.get_collection(name=COLLECTION_NAME)
    print(f"Successfully connected to RAG collection: {COLLECTION_NAME}")
except Exception as e:
    print(f"Error connecting to Chroma collection: {e}. Make sure you've run ingest_docs.py.")
    exit()

parler_model = None
parler_prompt_tokenizer = None
parler_description_tokenizer = None
try:
    print(f"Loading Indic Parler TTS model ({PARLER_MODEL_NAME}) and tokenizers... This may take a while.")
    parler_model = ParlerTTSForConditionalGeneration.from_pretrained(PARLER_MODEL_NAME).to(tts_device)
    parler_prompt_tokenizer = AutoTokenizer.from_pretrained(PARLER_MODEL_NAME)
    parler_description_tokenizer = AutoTokenizer.from_pretrained(parler_model.config.text_encoder._name_or_path)
    print("Indic Parler TTS loaded successfully.")
except Exception as e:
    print(f"Fatal Error loading Indic Parler TTS: {e}")
    import traceback
    traceback.print_exc()
    parler_model = None 
print("All components initialized (or attempted).")

# --- Helper Functions ---
def record_audio(filename=RECORDING_FILE, duration=DURATION, rate=SAMPLE_RATE, channels=CHANNELS):
    # (Same as your last version)
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
        return None, "unknown" # Return unknown for language
    if not os.path.exists(abs_whisper_model_path):
        print(f"Whisper model not found at: {abs_whisper_model_path}")
        return None, "unknown"

    expected_json_filename = f"{abs_audio_path}.json"
    if os.path.exists(expected_json_filename):
        try: os.remove(expected_json_filename)
        except Exception as e: print(f"Warning: Could not remove pre-existing JSON file {expected_json_filename}: {e}")

    command = [abs_whisper_executable, "-m", abs_whisper_model_path, "-f", abs_audio_path, "-l", "auto", "-oj"]
    print(f"Executing Whisper.cpp command: {' '.join(command)}")
    
    try:
        process = subprocess.run(command, capture_output=True, text=True, check=True, 
                                 cwd=os.path.dirname(abs_whisper_executable), encoding='utf-8')
        print("Whisper.cpp STDOUT:\n", process.stdout) # Contains plain text transcription
        print("Whisper.cpp STDERR:\n", process.stderr) # Contains logs, including internal lang detection

        if not os.path.exists(expected_json_filename):
            print(f"Error: Whisper.cpp did not create JSON file: {expected_json_filename}")
            # Fallback: Try to parse language from stderr if possible, and use stdout for text
            text_from_stdout = process.stdout.strip()
            lang_from_stderr = "unknown"
            for line in process.stderr.splitlines():
                if "auto-detected language:" in line:
                    try: lang_from_stderr = line.split("auto-detected language:")[1].split("(")[0].strip()
                    except: pass
                    break
            print(f"Fallback: Text from stdout: {text_from_stdout}, Lang from stderr: {lang_from_stderr}")
            return text_from_stdout, lang_from_stderr.split('-')[0].lower() if lang_from_stderr != "unknown" else "unknown"

        with open(expected_json_filename, 'r', encoding='utf-8') as f:
            transcription_json = json.load(f)
        
        try: os.remove(expected_json_filename)
        except Exception as e: print(f"Warning: Could not remove JSON file {expected_json_filename}: {e}")

        full_text = ""
        segments = transcription_json.get("transcription", [])
        if isinstance(segments, list):
            full_text = " ".join([segment.get("text", "").strip() for segment in segments]).strip()
        elif "text" in transcription_json:
            full_text = transcription_json["text"].strip()
        elif isinstance(transcription_json, list): # Some versions might output a list of segments directly
            full_text = " ".join([segment.get("text", "").strip() for segment in transcription_json]).strip()

        detected_language = "unknown"
        # Try to get language from the top-level "language" field in JSON
        lang_data_from_json = transcription_json.get("language")
        if isinstance(lang_data_from_json, dict):
            detected_language = lang_data_from_json.get("language", lang_data_from_json.get("lang", "unknown"))
        elif isinstance(lang_data_from_json, str): # If "language": "hi" directly
            detected_language = lang_data_from_json
        
        # If still unknown, parse from stderr as a last resort (already partially done if file not found)
        if detected_language == "unknown":
             for line in process.stderr.splitlines(): # Re-check stderr if JSON lacked lang
                if "auto-detected language:" in line:
                    try: detected_language = line.split("auto-detected language:")[1].split("(")[0].strip()
                    except: pass
                    break

        print(f"Transcription from JSON: {full_text}")
        print(f"Detected language (Whisper): {detected_language}")
        return full_text, detected_language.split('-')[0].lower() if detected_language != "unknown" else "unknown"

    except Exception as e:
        print(f"Error during transcription: {e}")
        import traceback
        traceback.print_exc()
        return None, "unknown"

def get_rag_context(query_text, top_k=RAG_TOP_K):
    # (Same as your last version)
    if not query_text: return ""
    query_embedding = embedding_model.encode(f"query: {query_text}").tolist()
    results = rag_collection.query(query_embeddings=[query_embedding], n_results=top_k)
    context = "\n".join(results['documents'][0]) if results['documents'] and results['documents'][0] else ""
    print(f"Retrieved context:\n---\n{context if context else 'No context found.'}\n---")
    return context

def query_llm(user_query, rag_context, input_lang="en"):
    # (Same as your last version)
    print(f"Querying LLM with input_lang hint: {input_lang}...")
    system_prompt = ("You are a helpful multilingual AI assistant...") # Keep your detailed prompt
    if input_lang == "hi": system_prompt += " Please respond primarily in clear Hindi."
    else: system_prompt += " Please respond primarily in clear English."
    prompt = f"Context:\n{rag_context if rag_context else 'N/A'}\n---\nUser Query: {user_query}\n\nAssistant Response:"
    try:
        response = ollama.chat(model=OLLAMA_MODEL, messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}])
        llm_response_text = response['message']['content']
        print(f"LLM Response: {llm_response_text}")
        return llm_response_text
    except Exception as e:
        print(f"Error querying Ollama: {e}")
        return "Sorry, error processing request."

def synthesize_speech_indic_parler(text_to_speak, lang_code="en"):
    if not parler_model:
        print("Indic Parler TTS model not loaded. Cannot synthesize.")
        return None, 24000  
    print(f"Synthesizing with Indic Parler: lang_code for desc: '{lang_code}', text: '{text_to_speak[:60]}...'")
    description = DESCRIPTION_DEFAULT
    if lang_code == "hi": description = DESCRIPTION_HI
    elif lang_code == "en": description = DESCRIPTION_EN
    
    audio_arr_int16 = None # Initialize
    tts_sample_rate = parler_model.config.sampling_rate if parler_model else 24000 # Default
    audio_bytes = None

    try:
        prompt_inputs = parler_prompt_tokenizer(text_to_speak, return_tensors="pt", padding=True, truncation=True, max_length=512)
        description_inputs = parler_description_tokenizer(description, return_tensors="pt", padding=True, truncation=True, max_length=128)
        generation = parler_model.generate(
            input_ids=description_inputs.input_ids.to(tts_device),
            attention_mask=description_inputs.attention_mask.to(tts_device),
            prompt_input_ids=prompt_inputs.input_ids.to(tts_device),
            prompt_attention_mask=prompt_inputs.attention_mask.to(tts_device),
            do_sample=True, temperature=0.6, top_k=50, top_p=0.95).cpu()
        
        audio_arr = generation.numpy().squeeze()
        if audio_arr.ndim > 1: audio_arr = audio_arr.flatten()
        if audio_arr.dtype == np.float32: audio_arr_int16 = (audio_arr * 32767).astype(np.int16)
        elif audio_arr.dtype == np.int16: audio_arr_int16 = audio_arr
        else: audio_arr_int16 = audio_arr.astype(np.int16)
        
        audio_bytes = audio_arr_int16.tobytes()
        tts_sample_rate = parler_model.config.sampling_rate
        
        # --- Save for debugging ---
        if audio_arr_int16 is not None: # Make sure we have the int16 array
            if not os.path.exists("audio_files"): os.makedirs("audio_files")
            debug_audio_path = os.path.join("audio_files", "debug_parler_tts_output.wav")
            print(f"Saving debug TTS audio to: {debug_audio_path} (SR: {tts_sample_rate})")
            sf.write(debug_audio_path, audio_arr_int16, samplerate=tts_sample_rate, subtype='PCM_16')
            print("Debug Parler TTS audio saved.")
        # --- End save for debugging ---

        print(f"Indic Parler TTS synthesis successful. Sample rate: {tts_sample_rate}")
        return audio_bytes, tts_sample_rate
    except Exception as e:
        print(f"Error during Indic Parler TTS synthesis: {e}")
        import traceback
        traceback.print_exc()
        return None, tts_sample_rate

def play_audio_bytes_simpleaudio(audio_bytes, sample_rate, sample_width=2, channels=CHANNELS):
    if not audio_bytes: print("No audio data to play (simpleaudio)."); return
    print(f"Playing with simpleaudio: {len(audio_bytes)} bytes, SR: {sample_rate} Hz")
    frame_size = sample_width * channels
    if len(audio_bytes) % frame_size != 0:
        print(f"Warning (simpleaudio): Audio data length ({len(audio_bytes)}) not multiple of frame size ({frame_size}).")
    try:
        sa.play_buffer(audio_bytes, num_channels=channels, bytes_per_sample=sample_width, sample_rate=sample_rate).wait_done()
        print("Playback with simpleaudio finished.")
    except Exception as e:
        print(f"Error during simpleaudio playback: {e}") # This is where segfault happens
        import traceback
        traceback.print_exc()

def play_audio_bytes_sounddevice(audio_bytes, sample_rate, channels=CHANNELS): # sample_width implied by dtype
    if not audio_bytes: print("No audio data to play (sounddevice)."); return
    print(f"Playing with sounddevice: {len(audio_bytes)} bytes, SR: {sample_rate} Hz")
    try:
        audio_np = np.frombuffer(audio_bytes, dtype=np.int16)
        if channels > 1 and audio_np.size % channels != 0 : # Reshape if stereo needed
             print(f"Warning (sounddevice): attempting to reshape for {channels} channels")
             audio_np = audio_np.reshape(-1, channels)
        sd.play(audio_np, samplerate=sample_rate, blocking=True)
        print("Playback with sounddevice finished.")
    except Exception as e:
        print(f"Error during sounddevice playback: {e}")
        import traceback
        traceback.print_exc()

# --- Main Loop ---
if __name__ == "__main__":
    if not os.path.exists(WHISPER_EXECUTABLE):
        print(f"Whisper.cpp executable not found: '{os.path.abspath(WHISPER_EXECUTABLE)}'")
        exit(1)
    if not parler_model:
        print("Indic Parler TTS model failed to load. Exiting.")
        exit(1)

    # Create audio_files directory if it doesn't exist
    if not os.path.exists("audio_files"):
        os.makedirs("audio_files")
        print("Created directory: audio_files")


    playback_method = "sounddevice" # "simpleaudio" or "sounddevice" - CHANGE THIS TO TEST

    try:
        while True:
            action = input("\nPress Enter to record, or type 'q' to quit: ")
            if action.lower() == 'q': break

            audio_file = record_audio()
            user_query_text, detected_lang_whisper = transcribe_audio_whisper_cpp(audio_file)

            if not user_query_text:
                print("Could not transcribe audio.")
                tts_err_text = "Sorry, I could not understand that."
                err_audio_bytes, err_sr = synthesize_speech_indic_parler(tts_err_text, lang_code="en")
                if err_audio_bytes:
                    if playback_method == "simpleaudio": play_audio_bytes_simpleaudio(err_audio_bytes, err_sr)
                    else: play_audio_bytes_sounddevice(err_audio_bytes, err_sr)
                continue
            
            primary_lang_code_for_llm = detected_lang_whisper # Already .lower() and primary from transcribe func

            rag_context = get_rag_context(user_query_text)
            llm_response = query_llm(user_query_text, rag_context, input_lang=primary_lang_code_for_llm)

            tts_speaker_lang_preference = "en" 
            if primary_lang_code_for_llm == "hi":
                tts_speaker_lang_preference = "hi"
            
            audio_output_bytes, tts_sr = synthesize_speech_indic_parler(llm_response, lang_code=tts_speaker_lang_preference)
            
            if audio_output_bytes:
                print(f"Attempting playback using: {playback_method}")
                if playback_method == "simpleaudio":
                    play_audio_bytes_simpleaudio(audio_output_bytes, tts_sr)
                else: # sounddevice
                    play_audio_bytes_sounddevice(audio_output_bytes, tts_sr)
            else:
                print("Could not synthesize speech for the response.")
                fb_tts_text = "I have a response, but encountered an issue speaking it."
                fb_audio, fb_sr = synthesize_speech_indic_parler(fb_tts_text, lang_code="en")
                if fb_audio:
                    if playback_method == "simpleaudio": play_audio_bytes_simpleaudio(fb_audio, fb_sr)
                    else: play_audio_bytes_sounddevice(fb_audio, fb_sr)
    except KeyboardInterrupt:
        print("\nExiting agent...")
    except Exception as e:
        print(f"Unexpected error in main loop: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if os.path.exists(RECORDING_FILE):
            try: os.remove(RECORDING_FILE)
            except Exception as e: print(f"Could not remove temp file {RECORDING_FILE}: {e}")
        print("Cleanup complete.")