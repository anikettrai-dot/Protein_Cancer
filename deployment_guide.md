# Deployment Guide: Protein Cancer Dashboard

This guide provides step-by-step instructions to push your project to GitHub and host it on Streamlit Cloud.

## 1. Push to GitHub

Since I cannot access your GitHub credentials, you need to perform the final push. Follow these steps:

1. **Initialize Git (if not done already)**:
   In your terminal at `D:\Protein_Cancer`, run:
   ```powershell
   git init
   ```

2. **Add Files**:
   I have already configured `.gitignore` to skip the 13GB dataset but include the models and test samples.
   ```powershell
   git add .
   ```

3. **Commit**:
   ```powershell
   git commit -m "Initial commit: Protein Cancer Dashboard with models and test samples"
   ```

4. **Create a Repository on GitHub**:
   - Go to [github.com](https://github.com) and create a new repository (e.g., `Protein-Cancer-Analysis`).
   - Do **not** initialize with a README, license, or gitignore.

5. **Connect and Push**:
   Replace `<YOUR_GITHUB_URL>` with your actual repository URL:
   ```powershell
   git remote add origin <YOUR_GITHUB_URL>
   git branch -M main
   git push -u origin main
   ```

> [!NOTE]
> If any model file is larger than 100MB, GitHub might reject the push. If that happens, let me know, and we will use Git LFS.

---

## 2. Host on Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io).
2. Sign in with your GitHub account.
3. Click **"Create app"**.
4. Select your repository (`Protein-Cancer-Analysis`) and the branch (`main`).
5. Set the **Main file path** to: `Protein_Project/app.py`.
6. Click **"Deploy!"**.

---

## 3. Testing the App

Once deployed:
1. Open the provided public URL.
2. In the sidebar, click the **"Upload PDB File"** area.
3. You can find test files in the `Protein_Project/test_samples` folder in your repository to verify it works.
