
from murf import Murf

client = Murf(api_key="ap2_1c084a20-f661-44b0-bec9-49a4646a2c3a")

response = client.text_to_speech.generate(
  text = "I am Guiding Your Child Towards Their Dream College Journey Meet with a College mentor advisor to......Read more at: https://www.collegementor.com/colleges",
  voice_id = "en-US-natalie"
)

print(response.audio_file)
