import requests
import torch
from transformers import pipeline

class AudioGenerator:
    def __init__(self, llm_model="gpt2", bark_model=None):
        self.llm = pipeline("text-generation", model=llm_model)
        if bark_model is None:
            from bark import SAMPLE_RATE, generate_audio, preload_models
            preload_models()
            self.bark_generate_audio = generate_audio
            self.sample_rate = SAMPLE_RATE
        else:
            self.bark_generate_audio = bark_model["generate_audio"]
            self.sample_rate = bark_model.get("SAMPLE_RATE", 24000)

    def generate_prompt(self, user_input, max_length=150):
        """Use LLM to generate a descriptive prompt for audio generation."""
        prompt = self.llm(user_input, max_length=max_length)[0]['generated_text']
        return prompt

    def text_to_audio(self, text_prompt, output_path="output.wav", style_preset="v2/en_speaker_6"):
        """Generate audio from the text prompt using Bark."""
        audio_array = self.bark_generate_audio(text_prompt, history_prompt=style_preset)
        # Save to output file
        import scipy.io.wavfile
        scipy.io.wavfile.write(output_path, self.sample_rate, audio_array)
        return output_path

    def from_user_input(self, user_input, output_path="output.wav"):
        prompt = self.generate_prompt(user_input)
        audio_file = self.text_to_audio(prompt, output_path=output_path)
        return audio_file, prompt

# Example usage
if __name__ == "__main__":
    ag = AudioGenerator()
    user_input = "Read this text in a cheerful tone: Hello, welcome to the audio generator!"
    audio_file, final_prompt = ag.from_user_input(user_input)
    print(f"Generated prompt: {final_prompt}")
    print(f"Audio saved at: {audio_file}")
