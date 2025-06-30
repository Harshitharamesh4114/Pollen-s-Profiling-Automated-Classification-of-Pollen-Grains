import os
import logging
from flask import Flask, render_template, request, flash, redirect, url_for, jsonify
from werkzeug.utils import secure_filename
from werkzeug.middleware.proxy_fix import ProxyFix
import cv2
import numpy as np

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Pollen classification classes
POLLEN_CLASSES = [
    'Anacardiaceae', 'Arecaceae', 'Asteraceae', 'Bignoniaceae',
    'Burseraceae', 'Cecropia', 'Combretaceae', 'Euphorbiaceae',
    'Fabaceae_Caesalpinioideae', 'Fabaceae_Faboideae', 'Fabaceae_Mimosoideae',
    'Lauraceae', 'Malpighiaceae', 'Malvaceae', 'Melastomataceae',
    'Moraceae', 'Myrtaceae', 'Palmae', 'Poaceae',
    'Proteaceae', 'Rubiaceae', 'Sapindaceae', 'Urticaceae'
]

class SimplePollenClassifier:
    """Simple pollen classifier for demonstration"""
    
    def __init__(self):
        self.classes = POLLEN_CLASSES
        self.img_size = 128
    
    def preprocess_image(self, image_path):
        """Preprocess image for prediction"""
        try:
            img = cv2.imread(image_path)
            if img is None:
                return None
            
            # Convert BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Resize to standard size
            img = cv2.resize(img, (self.img_size, self.img_size))
            
            # Normalize
            img = img.astype(np.float32) / 255.0
            
            return img
        except Exception as e:
            logging.error(f"Error preprocessing image: {e}")
            return None
    
    def predict(self, image_path):
        """Make prediction on uploaded image"""
        try:
            # Preprocess image
            processed_img = self.preprocess_image(image_path)
            if processed_img is None:
                return None
            
            # Simulate prediction based on image characteristics
            # In real implementation, this would use trained CNN model
            np.random.seed(int(np.sum(processed_img) * 1000) % 2147483647)
            predictions = np.random.random(len(self.classes))
            predictions = predictions / np.sum(predictions)
            
            # Get top prediction
            class_idx = np.argmax(predictions)
            confidence = float(predictions[class_idx])
            predicted_class = self.classes[class_idx]
            
            # Get top 3 predictions
            top_3_indices = np.argsort(predictions)[-3:][::-1]
            top_3_predictions = []
            
            for idx in top_3_indices:
                top_3_predictions.append({
                    'class': self.classes[idx],
                    'confidence': float(predictions[idx]) * 100
                })
            
            result = {
                'predicted_class': predicted_class,
                'confidence': confidence * 100,
                'top_3': top_3_predictions,
                'all_predictions': {self.classes[i]: float(predictions[i] * 100) 
                                  for i in range(len(self.classes))}
            }
            
            return result
            
        except Exception as e:
            logging.error(f"Error making prediction: {e}")
            return None

# Create the Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "pollen_classification_secret_key")
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)

# Configuration
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff'}
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize classifier
classifier = SimplePollenClassifier()

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def validate_image(file_path):
    """Validate that the uploaded file is a valid image"""
    try:
        img = cv2.imread(file_path)
        return img is not None
    except Exception:
        return False

@app.route('/')
def index():
    """Home page with upload interface"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle image upload and prediction"""
    if 'image' not in request.files:
        flash('No image file selected', 'error')
        return redirect(url_for('index'))
    
    file = request.files['image']
    if file.filename == '':
        flash('No image file selected', 'error')
        return redirect(url_for('index'))
    
    if not allowed_file(file.filename):
        flash('Invalid file type. Please upload PNG, JPG, JPEG, BMP, or TIFF files only.', 'error')
        return redirect(url_for('index'))
    
    try:
        # Save uploaded file
        filename = secure_filename(file.filename)
        timestamp = str(int(np.random.random() * 1000000))
        filename = f"{timestamp}_{filename}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Validate image
        if not validate_image(file_path):
            os.remove(file_path)
            flash('Invalid image file. Please upload a valid image.', 'error')
            return redirect(url_for('index'))
        
        # Make prediction
        prediction_result = classifier.predict(file_path)
        
        if prediction_result is None:
            os.remove(file_path)
            flash('Error processing image. Please try again.', 'error')
            return redirect(url_for('index'))
        
        return render_template('result.html', 
                             prediction=prediction_result,
                             image_path=f"uploads/{filename}")
    
    except Exception as e:
        app.logger.error(f"Error during prediction: {e}")
        if 'file_path' in locals() and os.path.exists(file_path):
            os.remove(file_path)
        flash('An error occurred while processing your image. Please try again.', 'error')
        return redirect(url_for('index'))

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'classifier': 'loaded'})

@app.errorhandler(413)
def too_large(e):
    flash('File too large. Please upload an image smaller than 16MB.', 'error')
    return redirect(url_for('index'))

@app.errorhandler(404)
def not_found(e):
    return render_template('index.html'), 404

@app.errorhandler(500)
def server_error(e):
    app.logger.error(f"Server error: {e}")
    flash('An internal error occurred. Please try again.', 'error')
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)