import re
import os
import json
import logging
import spacy
import cv2
import pytesseract
from pytesseract import Output
import fitz  # PyMuPDF
import docx
from docx.shared import Inches
from PIL import Image, ImageDraw, ImageFont
import requests

# --- INITIALIZATION (No changes here) ---
logger = logging.getLogger(__name__)
nlp = spacy.load("en_core_web_sm")
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-pro:generateContent"
REDACTION_TEXT = "[REDACTED]"
PII_PATTERNS = {
    'EMAIL': re.compile(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'),
    'PHONE': re.compile(r'(\+91[\-\s]?)?[0]?(91)?[789]\d{9}'),
    'CREDIT_CARD': re.compile(r'\b(?:\d[ -]*?){13,16}\b'),
    'AADHAAR': re.compile(r'\b\d{4}[ -]?\d{4}[ -]?\d{4}\b'),
    'PAN': re.compile(r'[A-Z]{5}[0-9]{4}[A-Z]{1}'),
}

# --- PII DETECTION LOGIC (No changes here) ---
# ... (all functions from detect_pii_with_gemini to find_all_pii remain the same)
def detect_pii_with_gemini(text, api_key):
    """Uses Google Gemini to detect PII in a block of text."""
    if not text or not text.strip():
        return []

    headers = {'Content-Type': 'application/json'}
    prompt = f"""
    Analyze the following text to identify any Personally Identifiable Information (PII).
    PII includes, but is not limited to: names, email addresses, phone numbers, physical addresses, credit card numbers, government ID numbers (like Aadhaar, PAN), dates of birth, etc.
    Return ONLY a single, flat JSON array containing strings of the exact PII data found. Do not include any explanations, introductory text, or formatting.
    Example response: ["John Doe", "john.doe@email.com", "9876543210"]

    Text to analyze:
    ---
    {text}
    ---
    """
    payload = json.dumps({"contents": [{"parts": [{"text": prompt}]}]})
    
    try:
        response = requests.post(f"{GEMINI_API_URL}?key={api_key}", headers=headers, data=payload, timeout=45)
        response.raise_for_status()
        
        result = response.json()
        content = result['candidates'][0]['content']['parts'][0]['text']
        
        json_str = content.strip().replace('`', '')
        if json_str.startswith('json'):
            json_str = json_str[4:].strip()

        detected_pii = json.loads(json_str)
        logger.info(f"Gemini detected {len(detected_pii)} PII elements.")
        return [str(item) for item in detected_pii]
    except (requests.RequestException, KeyError, json.JSONDecodeError) as e:
        logger.error(f"Gemini API call failed or returned invalid data: {e}")
        return []

def find_all_pii(text, options):
    """Combines Gemini, regex, and spaCy to find all PII."""
    all_pii_text = set()

    if options.get('gemini_api_key'):
        gemini_results = detect_pii_with_gemini(text, options['gemini_api_key'])
        for pii in gemini_results:
            all_pii_text.add(pii)

    for pii_type in options.get('pii_types', []):
        if pii_type in PII_PATTERNS:
            for match in PII_PATTERNS[pii_type].finditer(text):
                all_pii_text.add(match.group(0))

    if 'NAME' in options.get('pii_types', []):
        doc = nlp(text)
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                all_pii_text.add(ent.text)
    
    return sorted(list(all_pii_text), key=len, reverse=True)


# --- UTILITY FUNCTIONS (apply_watermark_to_image is unchanged) ---
def apply_watermark_to_image(image):
# ... (existing code for image watermarking)
    h, w, _ = image.shape
    overlay = image.copy()
    
    cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, image, 0.5, 0, image)

    font = cv2.FONT_HERSHEY_SIMPLEX
    text = "REDACTED"
    (text_w, text_h), _ = cv2.getTextSize(text, font, 2, 3)
    center_x, center_y = w // 2, h // 2
    
    M = cv2.getRotationMatrix2D((center_x, center_y), -45, 1.0)
    cos = abs(M[0, 0])
    sin = abs(M[0, 1])
    
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    
    M[0, 2] += (nW / 2) - center_x
    M[1, 2] += (nH / 2) - center_y
    
    rotated_text_canvas = cv2.warpAffine(image, M, (nW, nH), flags=cv2.INTER_LINEAR, borderValue=(0,0,0,0))
    cv2.putText(rotated_text_canvas, text, (center_x - text_w//2, center_y + text_h//2), font, 2, (255, 255, 255), 3, cv2.LINE_AA)

    M_inv = cv2.getRotationMatrix2D((nW // 2, nH // 2), 45, 1.0)
    M_inv[0, 2] += (w / 2) - nW//2
    M_inv[1, 2] += (h / 2) - nH//2
    
    final_img = cv2.warpAffine(rotated_text_canvas, M_inv, (w, h), flags=cv2.INTER_LINEAR, borderValue=(0,0,0,0))
    cv2.addWeighted(image, 1, final_img, 1, 0, image)
    return image


# --- *** NEW & IMPROVED PDF WATERMARK FUNCTION *** ---
def apply_pdf_watermark(page):
    """
    Applies a proper, centered, diagonal watermark to a PDF page using Shape.
    This is more robust than insert_text for rotated watermarks.
    """
    # Get page dimensions
    r = page.rect
    w = r.width
    h = r.height

    # Define the text properties for the watermark
    text = "REDACTED"
    font_size = min(w, h) / 10  # Dynamic font size
    text_len = fitz.get_text_length(text, fontname="helv-bold", fontsize=font_size)
    
    # Create a Shape for drawing
    shape = page.new_shape()

    # The point where we start drawing the text
    # We position it in the middle of the page
    pos = fitz.Point(w / 2 - text_len / 2, h / 2 + font_size / 4)

    # Define the text to be inserted
    shape.insert_text(
        pos,
        text,
        fontname="helv-bold",
        fontsize=font_size,
        color=(0.8, 0.8, 0.8),  # Light grey color
        fill_opacity=0.5,        # 50% transparent
        rotate=45,               # The desired 45-degree angle
    )
    
    # Apply the drawing to the page
    shape.commit()


# --- MAIN DISPATCHER (No changes here) ---
def process_file(file_path, options):
# ... (existing code)
    _, extension = os.path.splitext(file_path)
    extension = extension.lower()
    if extension in ['.png', '.jpg', '.jpeg']:
        return redact_image(file_path, options)
    elif extension == '.pdf':
        return redact_pdf(file_path, options)
    elif extension == '.docx':
        return redact_docx(file_path, options)
    elif extension == '.csv':
        return redact_csv(file_path, options)
    elif extension == '.json':
        return redact_json(file_path, options)
    else:
        raise ValueError(f"Unsupported file type: {extension}")

# --- FILE-SPECIFIC REDACTORS (redact_pdf is updated) ---

def redact_image(file_path, options):
# ... (existing code for redact_image)
    if options.get('wipe_metadata'):
        img_pil = Image.open(file_path)
        data = list(img_pil.getdata())
        image_without_metadata = Image.new(img_pil.mode, img_pil.size)
        image_without_metadata.putdata(data)
        image_without_metadata.save(file_path)

    img = cv2.imread(file_path)
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    for (x, y, w, h) in faces:
        if options.get('image_redaction_method') == 'blur':
            img[y:y+h, x:x+w] = cv2.GaussianBlur(img[y:y+h, x:x+w], (99, 99), 30)
        else:
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 0), -1)

    ocr_data = pytesseract.image_to_data(img, output_type=Output.DICT)
    full_text = " ".join(ocr_data['text'])
    pii_to_redact = find_all_pii(full_text, options)

    for pii in pii_to_redact:
        for word in pii.split():
            try:
                indices = [i for i, w in enumerate(ocr_data['text']) if w.lower() == word.lower()]
                for i in indices:
                    if float(ocr_data['conf'][i]) > 40:
                        (x, y, w, h) = (ocr_data['left'][i], ocr_data['top'][i], ocr_data['width'][i], ocr_data['height'][i])
                        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 0), -1)
            except ValueError:
                continue

    if options.get('apply_watermark'):
        img = apply_watermark_to_image(img)

    output_path = file_path.replace(os.path.splitext(file_path)[1], "_redacted.png")
    cv2.imwrite(output_path, img)
    return output_path

def redact_pdf(file_path, options):
    doc = fitz.open(file_path)
    if options.get('wipe_metadata'):
        doc.set_metadata({})

    for page_num, page in enumerate(doc):
        text = page.get_text("text")
        pii_to_redact = find_all_pii(text, options)
        
        for pii in pii_to_redact:
            areas = page.search_for(pii)
            for inst in areas:
                page.add_redact_annot(inst, fill=(0, 0, 0))
        
        page.apply_redactions()

        # *** THE FIX IS HERE ***
        # We now call our new, robust watermarking function.
        if options.get('apply_watermark'):
            apply_pdf_watermark(page)

    output_path = file_path.replace(".pdf", "_redacted.pdf")
    doc.save(output_path, garbage=4, deflate=True, clean=True)
    doc.close()
    return output_path

# ... (rest of the file: redact_docx, redact_csv, redact_json are unchanged)
def redact_docx(file_path, options):
    doc = docx.Document(file_path)
    full_text_content = "\n".join([p.text for p in doc.paragraphs])
    pii_to_redact = find_all_pii(full_text_content, options)

    new_doc = docx.Document()
    for para in doc.paragraphs:
        new_para_text = para.text
        for pii in pii_to_redact:
            if pii in new_para_text:
                new_para_text = new_para_text.replace(pii, REDACTION_TEXT)
        new_doc.add_paragraph(new_para_text)
    
    if options.get('wipe_metadata'):
        cp = new_doc.core_properties
        cp.author = "Redacted"
        cp.comments = "File processed and metadata wiped."
        cp.title = "Redacted Document"
        
    output_path = file_path.replace(".docx", "_redacted.docx")
    new_doc.save(output_path)
    return output_path

def redact_csv(file_path, options):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    pii_to_redact = find_all_pii(content, options)
    for pii in pii_to_redact:
        content = content.replace(pii, REDACTION_TEXT)

    output_path = file_path.replace(".csv", "_redacted.csv")
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(content)
    return output_path

def redact_json(file_path, options):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    pii_to_redact = find_all_pii(content, options)
    for pii in pii_to_redact:
        content = content.replace(pii, REDACTION_TEXT)

    output_path = file_path.replace(".json", "_redacted.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(content)
    return output_path

