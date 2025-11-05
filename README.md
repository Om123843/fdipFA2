# AquaVision – Water Sewage Detection

A full-stack web application that detects sewage contamination using **hybrid ML** (combining image processing + water quality parameters).

## 🔹 Tech Stack

- **Frontend**: React + Vite + TailwindCSS
- **Backend**: Flask (Python)
- **Model**: RandomForestClassifier (scikit-learn)
- **Image Processing**: OpenCV + Pillow

## 📁 Project Structure

```
WaterSewageDetection/
├── backend/
│   ├── app.py                    # Flask API
│   ├── train_model.py            # Train and save model
│   ├── model.pkl                 # (generated after training)
│   ├── water_quality_data.csv    # Sample dataset
│   ├── requirements.txt          # Python dependencies
│   └── static/uploads/           # Uploaded images
│
├── frontend/
│   ├── package.json
│   ├── index.html
│   ├── vite.config.js
│   ├── tailwind.config.cjs
│   ├── postcss.config.cjs
│   └── src/
│       ├── main.jsx
│       ├── App.jsx
│       ├── index.css
│       ├── pages/
│       │   └── Home.jsx
│       └── components/
│           ├── UploadForm.jsx
│           └── ResultCard.jsx
│
└── README.md
```

## 🚀 Quick Start

### 1️⃣ Backend Setup

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Train the model (creates model.pkl)
python train_model.py

# Start Flask server
python app.py
```

The backend will run on **http://localhost:5000**

### 2️⃣ Frontend Setup

Open a new terminal:

```powershell

cd frontend
npm install
npm run dev
```

The frontend will run on **http://localhost:5173** (or the port shown in terminal)

## 🎯 How It Works

1. **Upload**: User uploads a water sample image + enters sensor readings (pH, turbidity, DO, temperature, conductivity)
2. **Image Processing**: Backend extracts features:
   - Average RGB values
   - Blurriness (Laplacian variance)
   - Histogram spread
3. **Data Fusion**: Combines image features with sensor readings
4. **Prediction**: RandomForest model classifies as:
   - 🔴 **Sewage Detected** (contaminated)
   - 🟢 **Water is Clean**
5. **Result**: Shows prediction + confidence score

## 📊 Model Training

The `train_model.py` script:

- Creates a mock dataset if `water_quality_data.csv` doesn't exist
- Uses label rule: `if pH < 6.5 OR turbidity > 10 → Sewage (0), else Clean (1)`
- Trains RandomForestClassifier with 100 estimators
- Saves model to `model.pkl`

**To retrain with your own data:**

1. Replace `water_quality_data.csv` with your CSV containing columns:
   - `pH`, `turbidity`, `conductivity`, `DO`, `temperature`
   - `img_r`, `img_g`, `img_b`, `blur`, `hist_spread` (image features)
   - `label` (0=Sewage, 1=Clean)
2. Run `python train_model.py`

## 🔌 API Endpoint

**POST** `http://localhost:5000/predict`

**Request:**

- `Content-Type`: `multipart/form-data`
- Fields:
  - `image`: image file
  - `readings`: JSON string with sensor data
    ```json
    {
      "pH": 7.0,
      "turbidity": 5,
      "conductivity": 300,
      "DO": 6,
      "temperature": 22
    }
    ```

**Response:**

```json
{
  "prediction": "Sewage Detected",
  "confidence": 0.89
}
```

## 🛠️ Technologies Used

### Backend

- Flask - Web framework
- Flask-CORS - Cross-origin support
- OpenCV - Image processing
- scikit-learn - Machine learning
- joblib - Model serialization

### Frontend

- React 18 - UI framework
- Vite - Build tool
- TailwindCSS - Styling
- Axios - HTTP client

## 📝 Notes

- CSS lint warnings for `@tailwind` directives are expected - they're processed by PostCSS/Tailwind during build
- For production, use a proper WSGI server (e.g., Gunicorn) instead of Flask's dev server
- Add authentication and input validation for production use

## 📄 License

MIT

---

**Built with ❤️ for water quality monitoring**
