
import asyncio
import pandas as pd
import fasttext
import re

# Preprocess the text (example - adjust as needed)
def preprocess_text(text):
    token = str(text).lower()
    token = re.sub(r'[^\w\s]', '', token)  # Remove punctuation
    token = re.sub(r'\s+', ' ', token)  # Normalize whitespace (replace multiple spaces, tabs, newlines with a single space)
    token = token.strip() 
    return token


async def main():
    recipes_file = "./data/RAW_recipes.csv"
    reviews_file = "./data/RAW_interactions.csv"
    # Load the data
    recipes_df = pd.read_csv(recipes_file)
    reviews_df = pd.read_csv(reviews_file)

    # Combine relevant columns
    recipes_df['combined_text'] = recipes_df['description'] + ' ' + recipes_df['steps'] 
    all_text = pd.concat([recipes_df['combined_text'], reviews_df['review']])

    
    all_text = all_text.apply(preprocess_text)

    print(all_text.head())
    print(all_text.describe())
    print(len(all_text))

    traning_data_file = "./data/training_data.txt"
    model_bin_file = "./data/recipe_model.bin"
    model_txt_file = "./data/recipe_model.txt"
    # Save the combined text to a file
    with open(traning_data_file, "w") as f:
        for text in all_text:
            f.write(text + "\n")

    # # Train the FastText model
    model = fasttext.train_unsupervised(traning_data_file, model='skipgram')
    model.save_model(model_bin_file) 
    model_dim = model.get_dimension()

    print("Model dimension:", model_dim)

    with open(model_txt_file, "w") as f:
        words = model.words
        for word in words:
            #ensure its not an empty string
            word = preprocess_text(word)  # Clean the token
            if word:
                vector = model.get_word_vector(word)
                if(len(vector) == model_dim):
                    vector_str = " ".join([str(v) for v in vector])
                    vector_str = " ".join([str(v) for v in vector]) # More robust conversion to string
                    f.write(f"{word} {vector_str}\n") 

    

if __name__ == "__main__":
    asyncio.run(main())


    