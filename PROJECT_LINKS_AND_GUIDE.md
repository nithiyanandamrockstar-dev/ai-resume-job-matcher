# 🔗 AI Resume Analyzer & Job Matcher — Project Links & Guide

Saved Location: C:\Users\NIthi\Desktop\AI_Resume_Analyzer_Job_Matcher

---

## 🌐 Quick Access Links

- **Local Running App**: [http://localhost:8501](http://localhost:8501)
  *(Double-click RUN_APP.bat to launch the server anytime)*
- **Streamlit Community Cloud**: [https://share.streamlit.io](https://share.streamlit.io)
  *(Free 1-click cloud hosting platform for Streamlit)*
- **Create New GitHub Repo**: [https://github.com/new](https://github.com/new)
  *(Create public repository to connect to Streamlit Cloud)*
- **Official Python Download**: [https://www.python.org/downloads/](https://www.python.org/downloads/)
- **Streamlit Documentation**: [https://docs.streamlit.io/](https://docs.streamlit.io/)

---

## 🚀 How to Run the App Locally

### Option A: 1-Click Double Click
Double-click RUN_APP.bat inside this folder!

### Option B: PowerShell Terminal
`powershell
cd C:\Users\NIthi\Desktop\AI_Resume_Analyzer_Job_Matcher
& "C:\Users\NIthi\AppData\Local\Programs\Python\Python311\python.exe" -m streamlit run app.py
`

---

## ☁️ How to Push to GitHub & Deploy Online

`powershell
cd C:\Users\NIthi\Desktop\AI_Resume_Analyzer_Job_Matcher
git init
git add .
git commit -m "Deploy AI Resume Analyzer and Job Matcher"
git branch -M main
git remote add origin https://github.com/<YOUR-USERNAME>/<YOUR-REPO-NAME>.git
git push -u origin main
`
Then log in to [share.streamlit.io](https://share.streamlit.io), pick your repo and pp.py, and click **Deploy**!

---

## 📁 Project Contents in this Folder

- pp.py: Full AI NLP matching engine & Glassmorphic UI
- equirements.txt: Package dependencies for Streamlit Cloud
- .streamlit/config.toml: Dark mode theme & server configuration
- .gitignore: Git exclusion rules
- 	est_app.py: 7-scenario unit test suite
- data/sample_jobs.csv: Preloaded benchmark job descriptions
- data/samples/: Ready-to-test synthetic PDF resumes
- RUN_APP.bat: 1-click app launcher
