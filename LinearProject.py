import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import matplotlib.pyplot as plt

def calculate_pca_lsa(texts):
    # Step 1: TF-IDF vectorization
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(texts)

    # Step 2: PCA
    num_components = 2
    pca = PCA(n_components=num_components)
    pca_result = pca.fit_transform(tfidf_matrix.toarray())

    # Step 3: LSA (Truncated SVD)
    num_components_lsa = 2
    lsa = TruncatedSVD(n_components=num_components_lsa)
    lsa_result = lsa.fit_transform(tfidf_matrix)

    return pca_result, lsa_result

def display_pca_visualization(pca_result, texts):
    # Plotting
    fig, ax = plt.subplots(figsize=(8, 6))

    # Scatter plot
    ax.scatter(pca_result[:, 0], pca_result[:, 1])

    # Annotate each point with the corresponding text
    for i, txt in enumerate(texts):
        ax.annotate(txt, (pca_result[i, 0], pca_result[i, 1]))

    ax.set_title('PCA Visualization of Text Data')
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')

    # Show the plot
    st.pyplot(fig)

def display_similarity_analysis(input_word, paragraphs):
    # Step 4: Calculate cosine similarity
    similarity_scores = cosine_similarity([input_word], paragraphs)[0]

    # Display similarity scores
    st.write("Cosine Similarity Scores:", similarity_scores)

    # Plotting
    fig, ax = plt.subplots()
    ax.bar(range(len(similarity_scores)), similarity_scores * 100)
    ax.set_xticks(range(len(similarity_scores)))
    ax.set_xticklabels([f'Paragraph {i+1}' for i in range(len(similarity_scores))])
    ax.set_ylabel('Similarity Percentage')
    ax.set_title('Similarity Analysis')

    # Show the plot
    st.pyplot(fig)

def main():
    st.title("Word Similarity Analysis with PCA and LSA")

    # User inputs
    input_word = st.text_input("Enter a word for comparison:")
    input_paragraphs = st.text_area("Enter paragraphs (separate each paragraph with a newline):")

    # Submit button
    if st.button("Find Similarities"):
        # Process user inputs
        if input_word and input_paragraphs:
            paragraphs = input_paragraphs.split("\n")

            # Calculate PCA and LSA
            pca_result, lsa_result = calculate_pca_lsa([input_word] + paragraphs)

            # Display PCA result
            st.subheader("PCA Result:")
            st.write(pca_result)
            display_pca_visualization(pca_result, [input_word] + paragraphs)

            # Display LSA result
            st.subheader("LSA Result:")
            st.write(lsa_result)

            # Display Similarity Analysis
            st.subheader("Similarity Analysis:")
            display_similarity_analysis(lsa_result[0], lsa_result[1:])

if __name__ == "__main__":
    main()
