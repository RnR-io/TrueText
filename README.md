# TrueText: AI Essay Detector & Humanizer

TrueText is a powerful, single-file Python application built with Streamlit that helps you detect AI-generated content and humanize it to bypass detection.

## Features

-   **🕵️ AI Detection**: Uses the `roberta-base-openai-detector` model to analyze text and identify AI-generated sentences.
-   **📊 Visual Scoring**: Displays a dynamic gradient circle (Green to Red) indicating the probability of AI content.
-   **🖍️ Smart Highlighting**: Automatically highlights "Fake" (AI-generated) sentences in red for easy identification.
-   **✨ Humanizer**: Utilizes the `tuner007/pegasus_paraphrase` model to rewrite text and make it sound more natural.
-   **🚀 Fast & Efficient**: Caches models for quick loading and performance.

## Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/RnR-io/TrueText.git
    cd TrueText
    ```

2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

Run the application using Streamlit:

```bash
streamlit run app.py
```

1.  **Paste Text**: Enter your essay or text in the left input box.
2.  **Detect**: Click "Detect AI" to see the AI probability score and highlighted analysis.
3.  **Humanize**: Click "Humanize" to rewrite the text.

## Technologies

-   **Streamlit**: For the interactive web interface.
-   **Transformers (Hugging Face)**: For loading state-of-the-art NLP models.
-   **Torch**: Deep learning framework backend.
-   **NLTK**: For sentence tokenization.

## License

This project is open-source and available under the MIT License.
