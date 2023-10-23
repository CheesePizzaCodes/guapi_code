import requests
import json

API_KEY = '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'
ENDPOINT = 'https://api.openai.com/v1/engines/davinci-codex/completions'

headers = {
    'Content-Type': 'application/json',
    'Authorization': f'Bearer {API_KEY}',
}
def main(words):
    prompt = f'''
I will give you a list of 713 unique terms that refer to different manufacturing processes. I wan you to classify them in 4 categories:
-Fiberglass sandwich with foam core
-Fiberglass sandwich with wood core
-Fiberglass solid laminate
-Other (includes aluminum, steel, carbon fiber
-None (edge cases)
are you ready?
{words}
'''

    data = {
        'prompt': prompt,
        'max_tokens': 100,
        'n': 1,
        'stop': None,
        'temperature': 0.8,
    }

    response = requests.post(ENDPOINT, headers=headers, data=json.dumps(data))
    response_json = response.json()

    generated_text = response_json['choices'][0]['text']
    return generated_text

