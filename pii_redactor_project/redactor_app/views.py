import os
import json
import logging
from django.shortcuts import render
from django.http import JsonResponse, FileResponse
from django.views.decorators.csrf import csrf_exempt
from django.core.files.storage import FileSystemStorage
from . import services

logger = logging.getLogger(__name__)

# This helper class is from our previous fix and remains crucial.
class FileRemover:
    def __init__(self, file_path, mode='rb'):
        self.file_path = file_path
        self._file = open(file_path, mode)

    def __getattr__(self, attr):
        return getattr(self._file, attr)

    def close(self):
        self._file.close()
        if os.path.exists(self.file_path):
            os.remove(self.file_path)

def index(request):
    """Renders the main HTML page."""
    return render(request, 'index.html')

@csrf_exempt
def redact_file_view(request):
    if not (request.method == 'POST' and request.FILES.get('file')):
        return JsonResponse({'error': 'Invalid request'}, status=400)

    uploaded_file = request.FILES['file']
    
    # Collect all new options from the frontend
    # THE CHANGE IS HERE: We now check for an environment variable as a fallback.
    options = {
        'gemini_api_key': request.POST.get('gemini_api_key') or os.environ.get('GEMINI_API_KEY'),
        'pii_types': json.loads(request.POST.get('pii_types', '[]')),
        'wipe_metadata': request.POST.get('wipe_metadata') == 'true',
        'apply_watermark': request.POST.get('apply_watermark') == 'true',
        # Keep old options for image-specific redaction
        'image_redaction_method': request.POST.get('image_redaction_method', 'box'),
    }

    if not options.get('gemini_api_key'):
        return JsonResponse({'error': 'Gemini API Key is required for processing. Please enter it in the UI or set the GEMINI_API_KEY environment variable.'}, status=400)

    fs = FileSystemStorage()
    filename = fs.save(uploaded_file.name, uploaded_file)
    uploaded_file_path = fs.path(filename)
    
    try:
        redacted_file_path = services.process_file(uploaded_file_path, options)
        
        # We no longer include Supabase logic in this view for simplicity.
        # It can be re-added within the service layer if needed.

        response = FileResponse(FileRemover(redacted_file_path), as_attachment=True)
        return response

    except Exception as e:
        logger.error(f"Redaction failed for {filename}: {e}", exc_info=True)
        return JsonResponse({'error': f"An unexpected error occurred: {e}"}, status=500)
    
    finally:
        if os.path.exists(uploaded_file_path):
            os.remove(uploaded_file_path)


