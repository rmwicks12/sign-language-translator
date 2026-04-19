from django.shortcuts import render

def home(request):
    return render(request, 'translator/index.html')

def collect_data(request):
    return render(request, 'translator/collect.html')

import json
import os
import time
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

# @csrf_exempt allows us to send data locally without strict security tokens for now
@csrf_exempt
def save_dataset(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        word = data.get('word')
        frames = data.get('frames')

        # Create a 'dataset' folder in your project if it doesn't exist yet
        os.makedirs('dataset', exist_ok=True)

        # Create a unique filename using the word and a timestamp so nothing gets overwritten
        filename = f"dataset/{word}_{int(time.time())}.json"
        
        # Save the 30 frames of coordinates to the file
        with open(filename, 'w') as f:
            json.dump(frames, f)

        print(f"💾 Dataset saved: {filename}")
        return JsonResponse({'status': 'success', 'file': filename})
        
    return JsonResponse({'status': 'error'}, status=400)