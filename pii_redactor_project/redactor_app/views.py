import os
import tempfile
import logging
from django.http import JsonResponse, FileResponse
from django.views.decorators.csrf import csrf_exempt
from django.shortcuts import render
from . import services

logger = logging.getLogger(__name__)


# --- FUNCTION TO DISPLAY THE WEBPAGE ---
def index(request):
    """
    This view simply renders the main index.html page.
    """
    return render(request, 'index.html')

# --- HELPER CLASS FOR FILE DELETION ---

class FileRemover:
    def __init__(self, path):
        self.path = path
        self.file = open(self.path, 'rb')

    def __iter__(self):
        return self.file

    def close(self):
        self.file.close()
        try:
            os.remove(self.path)
            logger.info(f"Successfully removed temporary file: {self.path}")
        except OSError as e:
            logger.error(f"Error removing temporary file {self.path}: {e}")

@csrf_exempt
def redact_file_view(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'Only POST method is allowed'}, status=405)

    if 'file' not in request.FILES:
        return JsonResponse({'error': 'No file provided'}, status=400)

    uploaded_file = request.FILES['file']

    
    # Collect all new options from the frontend

    options = {
        'ai_model_choice': request.POST.get('ai_model_choice', 'local'),
        'pii_types': request.POST.getlist('pii_types[]'),
        'apply_watermark': request.POST.get('apply_watermark') == 'true',

        'wipe_metadata': request.POST.get('wipe_metadata') == 'true',
        'covert_redaction': request.POST.get('covert_redaction') == 'true',
        'gemini_api_key': os.getenv('GEMINI_API_KEY')

        'image_redaction_method': request.POST.get('image_redaction_method', 'box'),

    }

    uploaded_file_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{uploaded_file.name}") as temp_file:
            for chunk in uploaded_file.chunks():
                temp_file.write(chunk)
            uploaded_file_path = temp_file.name
        
        logger.info(f"File uploaded to temporary path: {uploaded_file_path}")
        redacted_file_path = services.process_file(uploaded_file_path, options)
        

        response = FileResponse(FileRemover(redacted_file_path))
        redacted_filename = os.path.basename(redacted_file_path)
        response['Content-Disposition'] = f'attachment; filename="{redacted_filename}"'
        response['Content-Type'] = 'application/octet-stream'

        logger.info(f"Sending redacted file '{redacted_filename}' to user.")

        response = FileResponse(FileRemover(redacted_file_path), as_attachment=True)

        return response

    except Exception as e:
        logger.error(f"Redaction failed for {uploaded_file.name}: {e}", exc_info=True)
        return JsonResponse({'error': f"Redaction failed: {e}"}, status=500)
    finally:
        if uploaded_file_path and os.path.exists(uploaded_file_path):
            try:
                os.remove(uploaded_file_path)
                logger.info(f"Successfully removed original uploaded file: {uploaded_file_path}")
            except OSError as e:
                logger.error(f"Error removing original uploaded file {uploaded_file_path}: {e}")

