from flask import Flask, request, send_file, jsonify
from PIL import Image
from enhance import enhance_image, denoise_image
import base64
from io import BytesIO

app = Flask(__name__)

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok"})

@app.route('/image/enhance', methods=['POST'])
def enhance():
    file = request.files['image']
    image = Image.open(file.stream)

    enhanced_image = enhance_image(image)
    enhanced_image = denoise_image(enhanced_image)

    img_io = BytesIO()
    enhanced_image.save(img_io, format='JPEG')
    img_io.seek(0)

    return send_file(img_io, mimetype='image/jpeg', as_attachment=True, download_name='enhanced.jpg')


@app.route('/image/enhance/batch', methods=['POST'])
def enhance_batch():
    files = request.files.getlist('images')

    enhanced_images = []

    for file in files:
        image = Image.open(file.stream)
        enhanced_image = enhance_image(image)
        enhanced_image = denoise_image(enhanced_image)
        buffered = BytesIO()
        enhanced_image.save(buffered, format='JPEG')
        img_b64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        enhanced_images.append(img_b64)

    return jsonify({"enhanced_images": enhanced_images})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)