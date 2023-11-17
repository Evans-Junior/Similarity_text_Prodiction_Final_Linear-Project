# Word Similarity Analysis with PCA and LSA

This Streamlit app allows users to analyze word similarity using Principal Component Analysis (PCA) and Latent Semantic Analysis (LSA).

## Overview

- **PCA Visualization:** Visualize the PCA of provided text data, showcasing the distribution of paragraphs in a reduced-dimensional space.

- **LSA Result:** Explore the result of Latent Semantic Analysis, revealing underlying topics in the entered paragraphs.

- **Similarity Analysis:** Find the similarity between a given word and provided paragraphs. The app calculates cosine similarity scores and displays the results in both numerical and graphical formats.

## How to Use

1. **Clone the Repository:**
    ```bash
    git clone https://github.com/Evans-Junior/Similarity_text_Prodiction_Final_Linear-Project
    cd word-similarity-analysis
    ```

2. **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3. **Run the App:**
    ```bash
    streamlit run app.py
    ```

4. **Access the App:**
    Open your web browser and navigate to [http://localhost:8501](http://localhost:8501).

5. **Interact with the App:**
    - Enter a word for comparison.
    - Input paragraphs, separating each paragraph with a newline.
    - Click the "Find Similarities" button to view results.

## Technologies Used

- Python
- Streamlit
- scikit-learn
- NumPy
- Matplotlib



## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


## Link to App

# https://similaritychecker.streamlit.app/