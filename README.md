---
title: Persplain
emoji: 📊
colorFrom: gray
colorTo: green
sdk: gradio
sdk_version: 3.24.1
app_file: app.py
pinned: false
---

# persplain
Explainable personality recognition using interpretable transformer learning models.

```
persplain/
├── .github/
│   └── workflows/
│       └── main.yml
├── .gitignore
├── app.py
├── data/
│   └── docs.csv (*)
├── notebook.ipynb
├── README.md
└── requirements.txt
```

* `app.py` - Gradio main file to be hosted on Hugging Face Spaces.
* `data/` - Contains the dataset used.
* `notebook.ipynb` - Jupyter notebook for training.
* `github/workflows/main.yml` - GitHub Actions workflow to transfer the app to Hugging Face Spaces.

`(*)` = Not included in the repository.