# 🚀 Quick Start Guide

## Get Started in 5 Minutes!

### Step 1: Set Up Environment
```bash
# Create virtual environment
python -m venv venv

# Activate it
source venv/bin/activate  # On macOS/Linux
# OR
venv\Scripts\activate  # On Windows

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Configure Kaggle API
```bash
# Download kaggle.json from kaggle.com/settings
# Move it to the right location
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

### Step 3: Download Dataset (Optional)
```bash
# List available datasets
python scripts/download_data.py --list

# Download default dataset
python scripts/download_data.py
```

### Step 4: Run the App!
```bash
streamlit run app.py
```

The app will open at `http://localhost:8501` 🎉

---

## 📝 What You Can Do

### Without Training a Model:
- ✅ Explore the dashboard
- ✅ View sample visualizations
- ✅ Interact with maps
- ✅ Analyze statistics
- ✅ Upload and view images

### With a Trained Model:
- ✅ Get real AI predictions
- ✅ Detect deforestation
- ✅ Risk assessment
- ✅ Generate reports

---

## 🎯 Next Steps

1. **Explore the App**: Navigate through all pages
2. **Upload Images**: Try the data analysis features
3. **Train a Model**: Run `python scripts/train_model.py`
4. **Make Predictions**: Use the AI Prediction page

---

## 🆘 Need Help?

- Check the main [README.md](README.md) for detailed instructions
- Review the [troubleshooting section](README.md#troubleshooting)
- Ensure all dependencies are installed

---

## 🎨 Features Highlights

- 🏠 **Dashboard**: Real-time metrics and trends
- 📊 **Data Analysis**: Upload and analyze satellite images
- 🤖 **AI Prediction**: ML-powered deforestation detection
- 🗺️ **Geospatial View**: Interactive maps with alert zones
- 📈 **Statistics**: Comprehensive analytics and reports

---

**Happy Monitoring! 🌳🌍**
