# Defect Validation App 🚀

An end-to-end Machine Learning + Flask project that classifies software defect summaries as **Valid** or **Invalid**, featuring a stylish, colorful UI with confidence progress bar and session statistics. This project demonstrates skills in ML, full-stack development, and user experience design with a practical application in software quality assurance.


## 📸 Demo
![Home Page](app/static/UI.png)
![Valid Input](app/static/Valid_defect.png)
![Invalid Input](app/static/Invalid_defect.png)


## 🔧 Tech Stack
Python, Flask, scikit-learn, pandas, joblib, HTML, CSS, JavaScript


## 🎯 Features
- ML pipeline: TF-IDF + SVD + Logistic Regression
- Flask API with `/predict` endpoint
- Stylish frontend with gradient background, animated buttons, and confidence progress bar
- Session stats showing Valid vs Invalid percentages
- Recruiter-ready project showcasing ML + full-stack skills


## 📂 Project Structure
defect-validation/
├── app/ (Flask backend + frontend)
├── model/ (training + preprocessing)
├── artifacts/ (saved ML models)
├── data/ (sample defect datasets)


## 🚀 Run Locally
Clone the project:
```bash
git clone https://github.com/YOUR_USERNAME/defect-validation-app.git
cd defect-validation-app


## Create virtual environment & install dependencies:
python -m venv .venv
.venv\Scripts\Activate.ps1   # Windows
source .venv/bin/activate    # Linux/Mac
pip install -r requirements.txt


## Start the Flask app :
python -m app.api


## Open in browser :
http://127.0.0.1:8000