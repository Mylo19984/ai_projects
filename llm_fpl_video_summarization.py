from langsmith import traceable
from langsmith.wrappers import wrap_anthropic
import anthropic
import yt_dlp
import whisper
import requests
import os
import getpass

LANGSMITH_ENDPOINT='https://api.smith.langchain.com'
LANGSMITH_PROJECT='fpl_summary'


def download_audio(youtube_url, output_path='audio'):
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': output_path,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'quiet': True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([youtube_url])

def transcribe_audio(audio_path):
    model = whisper.load_model("base")
    result = model.transcribe(audio_path)
    return result['text']

# Example usage

if not os.environ.get("ANTHROPIC_API_KEY"):
  os.environ["ANTHROPIC_API_KEY"] = getpass.getpass("Enter API key for Anthropic: ")
  API_KEY = os.getenv("ANTHROPIC_API_KEY")

video_id = 'PcV5y_0IXzI'
API_URL = "https://api.anthropic.com/v1/messages"
download_audio(f"https://www.youtube.com/watch?v={video_id}")
text = transcribe_audio('audio.mp3')
print(text)

client = wrap_anthropic(anthropic.Anthropic())

@traceable
def pipeline(user_input: str) -> str:
    response = client.messages.create(
        model="claude-3-5-sonnet-latest",  # change to your desired Anthropic model
        system="You are expert in premier league podcast.",
        messages=[{"role": "user", "content": f"Summarize the following text:\n{user_input}"}],
        max_tokens=700
    )
    return response.content

'''
@traceable
def summarize_with_claude(text):
    headers = {
        "x-api-key": API_KEY,
        "anthropic-version": "2023-06-01",
        "Content-Type": "application/json"
    }
    data = {
        "model": "claude-3-5-sonnet-latest",  # Use your preferred Claude 3 model
        "max_tokens": 700,
        "system": "You are expert in premier league podcast.",
        "messages": [
            {"role": "user", "content": f"Summarize the following text:\n{text}"}
        ]
    }
    response = requests.post(API_URL, headers=headers, json=data)
    response.raise_for_status()
    result = response.json()
    return result["content"][0]["text"]
'''

if __name__ == "__main__":
    input_text = text
    summary = pipeline(input_text)
    print("Summary:")
    print(summary)