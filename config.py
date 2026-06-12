import os


def _env_flag(name, default=False):
    return os.environ.get(name, '1' if default else '0').strip().lower() in (
        '1', 'true', 'yes', 'on',
    )


def _env_optional_bool(name):
    value = os.environ.get(name, 'auto').strip().lower()
    if value in ('1', 'true', 'yes', 'on'):
        return True
    if value in ('0', 'false', 'no', 'off'):
        return False
    return None

class Config:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
    MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB
    ALLOWED_EXTENSIONS = {'webp', 'png', 'jpg', 'jpeg', 'tiff', 'bmp'}

    # OCR settings — Russian + English for mixed Cyrillic/Latin text
    OCR_LANGUAGES = ['ru', 'en']
    # Which engine to use: 'hybrid' (Tesseract+EasyOCR, default), 'paddleocr'.
    # Override with the OCR_ENGINE environment variable.
    OCR_ENGINE = os.environ.get('OCR_ENGINE', 'hybrid').strip().lower()
    ENABLE_IMAGE_PREPROCESSING = True
    # False = 3 preprocess variants + dual EasyOCR (slow, best accuracy on CPU)
    # True  = 2 variants + single EasyOCR pass (~2-3x faster)
    OCR_FAST_MODE = _env_flag('OCR_FAST_MODE', default=True)
    # OCR_USE_GPU=true/false to force behavior, or "auto" to detect CUDA (default)
    OCR_USE_GPU = _env_optional_bool('OCR_USE_GPU')
    MAX_PROCESSING_TIME = 60  # seconds

    # Server
    HOST = os.environ.get('HOST', '0.0.0.0')
    PORT = int(os.environ.get('PORT', '5001'))
    DEBUG = _env_flag('DEBUG', default=False)

    # Registration card fields — English keys, Tajik labels on card
    FIELDS = {
        1: {
            'key': 'registration_card_number',
            'label': 'Registration Card Number',
            'tajik_keywords': [
                'рақами бақайдгирӣ', 'рақами қайд', 'бақайдгирӣ',
                'рақами корти', 'корти бақайдгирӣ', 'карточка регистрации',
                'registration card', 'варақаи бақайдгирӣ', 'варакаи',
            ],
            'description': 'Рақами корти бақайдгирӣ'
        },
        2: {
            'key': 'passport_number',
            'label': 'Passport Number',
            'tajik_keywords': [
                'шиноснома', 'шиносномаи', 'шиноснома №', 'паспорт',
                'рақами шиноснома', 'рақами паспорт', 'шиноснома но',
                'шиноснома н', 'шиносномано',
            ],
            'description': 'Рақами шиноснома'
        },
        3: {
            'key': 'citizenship',
            'label': 'Citizenship',
            'tajik_keywords': [
                'шаҳрвандӣ', 'шахрвандӣ', 'шаҳрванди', 'шахрванди',
                'гражданство', 'шахрванд', 'шаҳрванд',
            ],
            'description': 'Шаҳрвандӣ'
        },
        4: {
            'key': 'name_and_surname',
            'label': 'Name and Surname',
            'tajik_keywords': [
                'ном ва насаб', 'ному насаб', 'насаб', 'ном',
                'номи', 'насаби', 'фамилия', 'имя',
            ],
            'description': 'Ном ва насаб'
        },
        5: {
            'key': 'date_of_registration',
            'label': 'Date of Registration',
            'tajik_keywords': [
                'санаи бақайдгирӣ', 'санаи қайд', 'ба қайд гирифта',
                'ба кайд', 'ба қайд', 'санаи бакайдгирӣ',
            ],
            'description': 'Санаи бақайдгирӣ'
        },
        6: {
            'key': 'prs_mia_rt',
            'label': 'PRS MIA RT',
            'tajik_keywords': [
                'паспортӣ-бақайдгирии', 'паспортӣ бақайдгирии',
                'хшб', 'хшб вкд', 'паспортӣ', 'паспорти',
                'бақайдгирии вкд',
            ],
            'description': 'Паспортӣ-бақайдгирии ВКД ҶТ (PRS MIA RT)'
        },
        7: {
            'key': 'valid_until',
            'label': 'Valid Until',
            'tajik_keywords': [
                'то кай эътибор', 'эътибор', 'мӯҳлат', 'мухлат',
                'то кай', 'эътибор дорад', 'эъти6ор',
            ],
            'description': 'То кай эътибор дорад'
        },
        8: {
            'key': 'serial_control_number',
            'label': 'Serial/Control Number',
            'tajik_keywords': [
                'силсила', 'назорат', 'serial', 'силсилаи',
                'рақами силсила', 'рақами назорат',
            ],
            'description': 'Силсила / Рақами назорат'
        },
        9: {
            'key': 'mia',
            'label': 'MIA (Ministry of Internal Affairs)',
            'tajik_keywords': [
                'вазорати корҳои дохилӣ', 'вазорати корхои дохилӣ',
                'дохилӣ', 'дохили',
            ],
            'description': 'Вазорати корҳои дохилӣ (MIA)'
        },
        10: {
            'key': 'place_of_residence',
            'label': 'Place of Residence',
            'tajik_keywords': [
                'ҷои зист', 'чои зист', 'суроға', 'зист',
                'суроғаи', 'чои истиқомат', 'ҷои истиқомат',
                'чои истикомат', 'истиқомат', 'истикомат',
            ],
            'description': 'Ҷои зист'
        },
        11: {
            'key': 'place_of_residence_cont',
            'label': 'Place of Residence (continued)',
            'tajik_keywords': [
                'идома', 'давом', 'идомаи ҷои зист',
                'идомаи чои зист',
            ],
            'description': 'Идомаи ҷои зист'
        },
        12: {
            'key': 'inspector',
            'label': 'Inspector',
            'tajik_keywords': [
                'нозир', 'инспектор', 'нозири',
            ],
            'description': 'Нозир'
        },
        13: {
            'key': 'date_of_registration_extension',
            'label': 'Date of Registration/Extension',
            'tajik_keywords': [
                'тамдид', 'тамдиди бақайдгирӣ', 'санаи тамдид',
                'тамдиди', 'санаи бақайдгирӣ ё тамдид',
            ],
            'description': 'Санаи бақайдгирӣ ё тамдиди бақайдгирӣ'
        }
    }

    # Regex patterns for various field types
    # Try strict patterns first, then fall back to flexible ones for OCR noise
    DATE_PATTERNS = [
        r'\d{2}\.\d{2}\.\d{4}',    # DD.MM.YYYY
        r'\d{2}/\d{2}/\d{4}',      # DD/MM/YYYY
        r'\d{2}\.\d{2}\.\d{2}\b',  # DD.MM.YY
        r'\d{4}\.\d{2}\.\d{2}',    # YYYY.MM.DD
        r'\d{4}-\d{2}-\d{2}',      # YYYY-MM-DD
        # Flexible patterns for OCR noise (spaces, apostrophes instead of dots)
        r"\d{2}[' ]\d{2}[' ]\d{4}",     # DD'MM'YYYY or DD MM YYYY
        r"\d{2}[' ]\d{2}[' ]\d{2}\b",   # DD'MM'YY or DD MM YY
        r',?\d{2}\.\d{2}\.\d{4}',       # optional leading comma + DD.MM.YYYY
        # Very flexible: handle single-digit day/month, comma/space separators
        r'[^0-9]*\d{1,2}[.,/\'\s]+\d{1,2}[.,/\'\s]+\d{4}',
    ]

    # More permissive: allow trailing dash, handle OCR noise
    # Registration card number: pure digits (6-10 chars), unanchored to find within larger text
    # Examples: 1245183, №1245183 after stripping prefix
    REGISTRATION_CARD_PATTERN = r'\d{6,10}'
    # Passport number: letters followed by digits (e.g., AA1234567, EC567165)
    PASSPORT_NUMBER_PATTERN = r'^[A-Z]{1,3}\s?[CcСс]?\s?\d{6,8}$'
    # Serial/Control number: standalone 4-6 digit number (not part of longer number)
    # Uses negative lookahead/lookbehind to avoid matching substrings of longer digits
    SERIAL_NUMBER_NUMERIC_PATTERN = r'(?<!\d)\d{4,6}(?!\d)'

