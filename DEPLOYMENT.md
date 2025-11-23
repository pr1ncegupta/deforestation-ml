# Deployment Guide - Deforestation Monitoring System

This guide covers deploying the Satellite Deforestation Monitoring application to various platforms and preparing for production.

---

## 📋 Table of Contents

- [Prerequisites](#prerequisites)
- [Environment Setup](#environment-setup)
- [Local Deployment](#local-deployment)
- [GitHub Deployment](#github-deployment)
- [Cloud Deployment Options](#cloud-deployment-options)
- [Docker Deployment](#docker-deployment)
- [Troubleshooting](#troubleshooting)

---

## ✅ Prerequisites

Before deploying, ensure you have:

- Python 3.8 or higher
- Git installed and configured
- Kaggle account with API credentials
- (Optional) Docker for containerized deployment
- (Optional) Cloud platform account (Streamlit Cloud, Heroku, etc.)

---

## 🔐 Environment Setup

### Step 1: Secure Your Credentials

**CRITICAL**: Never commit sensitive credentials to Git!

1. **Copy the environment template:**
   ```bash
   cp .env.example .env
   ```

2. **Configure your Kaggle API credentials:**
   - Visit https://www.kaggle.com/settings
   - Create a new API token
   - Download `kaggle.json`
   - Copy credentials to `.env` file

3. **Verify `.env` is gitignored:**
   ```bash
   git check-ignore .env
   # Should output: .env
   ```

### Step 2: Verify Security

Run these commands to ensure sensitive files are protected:

```bash
# Check what would be committed
git status

# Verify sensitive files are ignored
git check-ignore -v .env kaggle.json data/models/*.h5

# Review .gitignore patterns
cat .gitignore
```

**Common files that MUST be ignored:**
- `.env` - Environment variables and API keys
- `kaggle.json` - Kaggle credentials
- `data/raw/*` - Downloaded datasets (can be large)
- `data/models/*.h5` - Trained models (large files)
- `__pycache__/` - Python cache files

---

## 💻 Local Deployment

### Quick Start

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/sustainable-etp.git
   cd sustainable-etp
   ```

2. **Create virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Setup environment:**
   ```bash
   cp .env.example .env
   # Edit .env with your credentials
   ```

5. **Run the application:**
   ```bash
   streamlit run app.py
   ```

6. **Access the app:**
   - Open browser to `http://localhost:8501`

---

## 🐙 GitHub Deployment

### Initial Repository Setup

1. **Initialize Git (if not already):**
   ```bash
   git init
   ```

2. **Review what will be committed:**
   ```bash
   git status
   git diff
   ```

3. **Verify no sensitive data:**
   ```bash
   # This should show nothing sensitive
   git ls-files
   
   # Double-check .env is NOT listed
   git ls-files | grep .env
   ```

4. **Create first commit:**
   ```bash
   git add .
   git commit -m "Initial commit: Deforestation monitoring system"
   ```

5. **Create GitHub repository:**
   - Go to https://github.com/new
   - Create new repository (don't initialize with README)
   - Copy the repository URL

6. **Push to GitHub:**
   ```bash
   git remote add origin https://github.com/yourusername/your-repo-name.git
   git branch -M main
   git push -u origin main
   ```

### Repository Settings

After pushing, configure your GitHub repository:

1. **Add repository description and topics:**
   - Topics: `machine-learning`, `deforestation`, `satellite-imagery`, `streamlit`, `environmental-monitoring`

2. **Enable GitHub Actions (optional):**
   - Set up CI/CD for automated testing

3. **Add secrets for deployment:**
   - Go to Settings → Secrets and variables → Actions
   - Add: `KAGGLE_USERNAME`, `KAGGLE_KEY`

---

## ☁️ Cloud Deployment Options

### Option 1: Streamlit Cloud (Recommended)

**Pros:** Free tier, easy setup, no server management

1. **Prepare the app:**
   - Ensure `requirements.txt` is up to date
   - Create `packages.txt` if system dependencies needed

2. **Deploy to Streamlit Cloud:**
   - Visit https://share.streamlit.io
   - Connect your GitHub account
   - Select your repository
   - Choose branch (main) and file (app.py)
   - Click "Deploy"

3. **Configure secrets:**
   - In Streamlit Cloud dashboard, go to app settings
   - Add secrets in TOML format:
     ```toml
     KAGGLE_USERNAME = "your_username"
     KAGGLE_KEY = "your_api_key"
     ```

### Option 2: Heroku

1. **Create `Procfile`:**
   ```
   web: streamlit run app.py --server.port=$PORT
   ```

2. **Deploy:**
   ```bash
   heroku create your-app-name
   heroku config:set KAGGLE_USERNAME=your_username
   heroku config:set KAGGLE_KEY=your_key
   git push heroku main
   ```

### Option 3: AWS/GCP/Azure

For enterprise deployments, refer to platform-specific documentation for:
- EC2/Compute Engine/VM instances
- Container services (ECS, Cloud Run, AKS)
- Serverless options (Lambda, Cloud Functions, Azure Functions)

---

## 🐳 Docker Deployment

### Build and Run

1. **Build Docker image:**
   ```bash
   docker build -t deforestation-monitor .
   ```

2. **Run container:**
   ```bash
   docker run -p 8501:8501 \
     -e KAGGLE_USERNAME=your_username \
     -e KAGGLE_KEY=your_key \
     deforestation-monitor
   ```

3. **Using docker-compose:**
   ```bash
   # Edit docker-compose.yml with your environment
   docker-compose up -d
   ```

### Production Considerations

- Use environment-specific Docker images
- Implement health checks
- Set up proper logging
- Use secrets management (Docker secrets, HashiCorp Vault)

---

## 🔧 Troubleshooting

### Common Issues

#### 1. Kaggle Authentication Error

**Error:** `OSError: Could not find kaggle.json`

**Solution:**
```bash
# Option A: Use environment variables
export KAGGLE_USERNAME=your_username
export KAGGLE_KEY=your_key

# Option B: Place kaggle.json in correct location
mkdir -p ~/.kaggle
cp kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

#### 2. Module Not Found Errors

**Error:** `ModuleNotFoundError: No module named 'streamlit'`

**Solution:**
```bash
# Ensure virtual environment is activated
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Reinstall dependencies
pip install -r requirements.txt
```

#### 3. Port Already in Use

**Error:** `OSError: [Errno 48] Address already in use`

**Solution:**
```bash
# Use different port
streamlit run app.py --server.port 8502

# Or kill process on port 8501
lsof -ti:8501 | xargs kill -9
```

#### 4. Git Shows .env as Modified

**Problem:** `.env` appears in `git status` but should be ignored

**Solution:**
```bash
# If .env was previously committed, remove from git (keep local)
git rm --cached .env

# Verify .gitignore includes .env
echo ".env" >> .gitignore

# Commit the changes
git add .gitignore
git commit -m "Remove .env from tracking"
```

#### 5. Large Files Rejected by GitHub

**Error:** `remote: error: File too large`

**Solution:**
```bash
# Add large files to .gitignore
echo "data/raw/*" >> .gitignore
echo "data/models/*.h5" >> .gitignore

# If already committed, use git filter-branch or BFG Repo-Cleaner
# Or use Git LFS for large model files
git lfs track "*.h5"
```

#### 6. Missing Model File in Production

**Problem:** Model file not available in deployed app

**Solutions:**
- Download model on startup (if hosted remotely)
- Build model as part of Docker image (increases image size)
- Use model versioning service (MLflow, DVC)
- Store model in cloud storage (S3, GCS) and download on demand

---

## 📊 Deployment Checklist

Before deploying to production:

- [ ] All sensitive credentials removed from code
- [ ] `.env.example` template created and documented
- [ ] `.gitignore` properly configured
- [ ] `requirements.txt` is complete and pinned
- [ ] Environment variables configured in deployment platform
- [ ] Database connections secured (if applicable)
- [ ] API rate limits configured
- [ ] Error logging and monitoring set up
- [ ] Health check endpoint implemented
- [ ] HTTPS/SSL configured
- [ ] Backup strategy in place
- [ ] Documentation updated
- [ ] Security audit completed

---

## 📚 Additional Resources

- [Streamlit Deployment Guide](https://docs.streamlit.io/streamlit-community-cloud/get-started/deploy-an-app)
- [Kaggle API Documentation](https://www.kaggle.com/docs/api)
- [Git Security Best Practices](https://docs.github.com/en/code-security/getting-started/best-practices-for-preventing-data-leaks-in-your-organization)
- [Docker Best Practices](https://docs.docker.com/develop/dev-best-practices/)

---

## 🆘 Getting Help

If you encounter issues:

1. Check the [Troubleshooting](#troubleshooting) section
2. Review application logs
3. Search existing GitHub issues
4. Open a new issue with:
   - Error message
   - Steps to reproduce
   - Environment details
   - Relevant logs (remove sensitive info!)

---

**Last Updated:** 2024-11-24
