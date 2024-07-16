import cv2
import pytesseract
import re

# Set the path to the Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Define fixed prices for items
item_prices = {
    'CODEAA': 20,
    'CODEBB': 30
}

def preprocess_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    denoised = cv2.medianBlur(binary, 3)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    morph = cv2.morphologyEx(denoised, cv2.MORPH_CLOSE, kernel)
    return morph

def extract_text_and_parse(frame):
    preprocessed = preprocess_frame(frame)
    custom_config = r'--oem 3 --psm 6'
    data = pytesseract.image_to_data(preprocessed, output_type=pytesseract.Output.DICT, config=custom_config)
    
    detected_items = []

    n_boxes = len(data['level'])
    for i in range(n_boxes):
        if int(data['conf'][i]) > 60:  # Filter out low confidence detections
            if 'text' in data and data['text'][i].strip():  # Ensure there is text
                text = data['text'][i].strip()
                detected_items.append(text)

    return detected_items

def calculate_total(items):
    total = 0.0
    for item in items:
        if item in item_prices:
            total += item_prices[item]
    return total

cap = cv2.VideoCapture(0)

items_scanned = []
final_bill = 0.0
finished = False

while not finished:
    ret, frame = cap.read()
    if not ret:
        break

    detected_items = extract_text_and_parse(frame)

    for item in detected_items:
        if item in item_prices and item not in items_scanned:
            items_scanned.append(item)
            print(f'Item scanned: {item}')
            if len(items_scanned) < len(item_prices):
                response = input('Scan another item? (y/n): ')
                if response.lower() != 'y':
                    finished = True
                    break
            else:
                finished = True
                break

    final_bill = calculate_total(items_scanned)
    print(f'Items scanned: {items_scanned}')
    print(f'Final Bill: ${final_bill}')

    cv2.imshow('Live Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
