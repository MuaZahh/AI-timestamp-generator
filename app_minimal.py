from flask import Flask, jsonify
import os
from datetime import datetime

app = Flask(__name__)

@app.route('/')
def home():
    return jsonify({
        'status': 'success',
        'message': 'AI Timestamp Generator is working!',
        'timestamp': datetime.utcnow().isoformat(),
        'environment': 'vercel' if os.getenv('VERCEL') else 'local'
    })

@app.route('/test')
def test():
    return jsonify({
        'status': 'success',
        'message': 'Test endpoint working!',
        'vercel': bool(os.getenv('VERCEL')),
        'python_version': os.sys.version
    })

@app.route('/api')
def api():
    return jsonify({
        'message': 'AI Timestamp Generator API',
        'version': '1.0.0',
        'status': 'minimal mode',
        'endpoints': ['/', '/test', '/api']
    })

if __name__ == '__main__':
    app.run(debug=True)

# For Vercel
def handler(request):
    return app(request.environ, lambda *args: None)