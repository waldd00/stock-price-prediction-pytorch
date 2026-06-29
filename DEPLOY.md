# Deploy

## Hugging Face Spaces (önerilen, torch için bol kaynak)
1. huggingface.co > New Space > SDK: Streamlit, Hardware: CPU basic (free)
2. Space reposuna proje dosyalarını yükle (git push veya web arayüzü)
3. Space README'sinin en üstüne şu front matter'ı koy:

   ---
   title: Stock Price Prediction
   emoji: chart
   colorFrom: blue
   colorTo: green
   sdk: streamlit
   sdk_version: 1.30.0
   app_file: app/streamlit_app.py
   pinned: false
   ---

4. Build bitince link: https://huggingface.co/spaces/KULLANICI/space-adi

## Streamlit Community Cloud
1. Repoyu GitHub'a public push et
2. share.streamlit.io > GitHub ile giriş > New app
3. Repo, branch: main, main file path: app/streamlit_app.py > Deploy
4. Link: https://APP-ADI.streamlit.app

## English

### Hugging Face Spaces (recommended, generous free CPU for torch)
1. huggingface.co > New Space > SDK: Streamlit, Hardware: CPU basic (free)
2. Upload the project files to the Space repo (git push or web UI)
3. Add this front matter to the top of the Space README:

   ---
   title: Stock Price Prediction
   emoji: chart
   colorFrom: blue
   colorTo: green
   sdk: streamlit
   sdk_version: 1.30.0
   app_file: app/streamlit_app.py
   pinned: false
   ---

4. Once the build finishes, the link is: https://huggingface.co/spaces/USERNAME/space-name

### Streamlit Community Cloud
1. Push the repo to GitHub as public
2. share.streamlit.io > Sign in with GitHub > New app
3. Repo, branch: main, main file path: app/streamlit_app.py > Deploy
4. Link: https://APP-NAME.streamlit.app
