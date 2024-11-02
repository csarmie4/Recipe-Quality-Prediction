import pandas as pd

def load_data(filepath):
    """Load the dataset from the specified CSV file."""
    return pd.read_csv(filepath)

def preprocess_data(df):
    """Preprocess the dataset."""
    # Convert data types
    df['RecipeServings'] = df['RecipeServings'].astype(int)
    df['HighScore'] = df['HighScore'].astype(int)

    # Convert CholesterolContent from milligrams to grams
    df['CholesterolContent'] = df['CholesterolContent'] / 1000.0

    # Drop rows with missing RecipeCategory
    df.dropna(subset=["RecipeCategory"], inplace=True)

    # Remove observations with zeros in most numeric features
    zero_observations = (df["Calories"] == 0) & (df["CholesterolContent"] == 0) & \
                        (df["ProteinContent"] == 0) & (df["CarbohydrateContent"] == 0) & \
                        (df["SugarContent"] == 0)

    df.drop(df.loc[zero_observations].index, inplace=True)
    
    # Remove duplicates
    df.drop_duplicates(inplace=True)

    # Reduce number of categories in RecipeCategory
    category_replacements = {
        'Clear Soup': 'Soup', 'Ham And Bean Soup': 'Soup', 'Black Bean Soup': 'Soup',
        'Mushroom Soup': 'Soup', 'Low Protein': 'Low of', 'Very Low Carbs': 'Low of',
        # Add all other replacements here
    }
    # df["RecipeCategory"].replace(category_replacements, inplace=True)
    df["RecipeCategory"] = df["RecipeCategory"].replace(category_replacements)


    return df
