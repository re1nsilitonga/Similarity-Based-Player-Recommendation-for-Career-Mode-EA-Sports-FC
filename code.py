# main.py
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.spatial.distance import pdist, squareform
import numpy as np

class PlayerRecommender:
    """
    A class to build a player recommendation system for EA Sports FC 25.

    This system uses player attributes to find similar players based on
    Euclidean distance after normalization and K-Means clustering.
    """

    def __init__(self, data_path, n_clusters=10, random_state=42):
        """
        Initializes the PlayerRecommender.

        Args:
            data_path (str): The file path for the player dataset (CSV).
            n_clusters (int): The number of clusters for K-Means.
            random_state (int): Random state for reproducibility of clustering.
        """
        self.data_path = data_path
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.df = None
        self.scaled_attributes = None
        self.distance_matrix = None
        self.player_names = None
        self.player_indices = None
        self._load_and_preprocess_data()
        self._calculate_similarity()
        self._cluster_players()

    def _load_and_preprocess_data(self):
        """
        Loads the dataset and preprocesses it.

        This involves selecting relevant attributes, handling potential missing
        values, and scaling the data.
        """
        try:
            # Load the dataset
            self.df = pd.read_csv(self.data_path)
        except FileNotFoundError:
            print(f"Error: The file '{self.data_path}' was not found.")
            # Create a dummy dataframe to avoid further errors if file not found
            self.df = pd.DataFrame({
                'Name': ['Messi', 'Ronaldo', 'Mbappe', 'De Bruyne', 'Haaland'],
                'OVR': [93, 92, 91, 91, 91],
                'PAC': [85, 89, 97, 76, 89],
                'SHO': [92, 93, 89, 86, 91],
                'PAS': [91, 81, 80, 93, 65],
                'DRI': [95, 87, 92, 86, 80],
                'DEF': [34, 34, 36, 61, 45],
                'PHY': [64, 75, 76, 78, 88]
            })
            print("Using a dummy dataset for demonstration.")


        # Define the core attributes to be used for similarity calculation
        # These attributes cover all major aspects of a player's ability
        attributes = [
            'Acceleration', 'Sprint Speed', 'Positioning', 'Finishing', 'Shot Power',
            'Long Shots', 'Volleys', 'Penalties', 'Vision', 'Crossing', 'Free Kick Accuracy',
            'Short Passing', 'Long Passing', 'Curve', 'Agility', 'Balance', 'Reactions',
            'Ball Control', 'Dribbling', 'Composure', 'Interceptions', 'Heading Accuracy',
            'Def Awareness', 'Standing Tackle', 'Sliding Tackle', 'Jumping', 'Stamina',
            'Strength', 'Aggression'
        ]

        # For the dummy dataset, we'll use a smaller set of attributes
        if 'OVR' in self.df.columns and 'Acceleration' not in self.df.columns:
            attributes = ['OVR', 'PAC', 'SHO', 'PAS', 'DRI', 'DEF', 'PHY']

        # Filter out goalkeepers and select only the relevant attribute columns
        if 'Position' in self.df.columns:
            self.df = self.df[self.df['Position'] != 'GK'].copy()

        # Fill any potential missing values with the median of the column
        for col in attributes:
             if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
                self.df[col].fillna(self.df[col].median(), inplace=True)
             else:
                print(f"Warning: Attribute '{col}' not found in the dataset. It will be ignored.")


        # Ensure all required attributes exist before proceeding
        final_attributes = [attr for attr in attributes if attr in self.df.columns]
        if not final_attributes:
            raise ValueError("None of the specified attributes were found in the dataset.")


        # Store player names for later retrieval
        self.player_names = self.df['Name'].copy()
        # Create a mapping from player name to index for quick lookups
        self.player_indices = pd.Series(self.df.index, index=self.df['Name'])

        # Normalize the attributes using StandardScaler
        scaler = StandardScaler()
        self.scaled_attributes = scaler.fit_transform(self.df[final_attributes])

    def _calculate_similarity(self):
        """
        Calculates the similarity between all pairs of players.

        This method computes the Euclidean distance between the normalized
        attribute vectors of the players and creates a square distance matrix.
        """
        # pdist calculates the condensed distance matrix (upper triangle)
        condensed_distance_matrix = pdist(self.scaled_attributes, 'euclidean')
        # squareform converts it into a full, symmetric distance matrix
        self.distance_matrix = squareform(condensed_distance_matrix)
        print("Similarity (distance matrix) calculated successfully.")

    def _cluster_players(self):
        """
        Groups players into clusters based on their attributes.

        Uses the K-Means algorithm on the scaled attributes to categorize
        players into distinct groups. The cluster labels are added to the main dataframe.
        """
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state, n_init=10)
        # We fit on the scaled attributes, as K-Means also relies on distance
        self.df['Cluster'] = kmeans.fit_predict(self.scaled_attributes)
        print(f"Players clustered into {self.n_clusters} groups successfully.")

    def get_recommendations(self, player_name, k=5):
        """
        Provides k similar player recommendations for a given target player.

        Args:
            player_name (str): The name of the player to get recommendations for.
            k (int): The number of similar players to recommend.

        Returns:
            pandas.DataFrame: A DataFrame containing the top k recommended players,
                              sorted by similarity (distance).
        """
        if player_name not in self.player_indices:
            # Find the closest match to the provided player name
            from difflib import get_close_matches
            matches = get_close_matches(player_name, self.player_names, n=1, cutoff=0.6)
            if not matches:
                return f"Player '{player_name}' not found in the dataset."
            player_name = matches[0]
            print(f"Player not found. Did you mean '{player_name}'?")


        # 1. Get the index and cluster of the target player
        player_idx = self.player_indices[player_name]
        player_cluster = self.df.loc[player_idx, 'Cluster']

        # 2. Get the distances from the target player to all other players
        player_distances = self.distance_matrix[player_idx]

        # 3. Create a DataFrame with player names, distances, and clusters
        recommendations_df = pd.DataFrame({
            'Name': self.player_names,
            'Distance': player_distances,
            'Cluster': self.df['Cluster']
        })

        # 4. Filter for players in the same cluster, excluding the player themselves
        cluster_recommendations = recommendations_df[
            (recommendations_df['Cluster'] == player_cluster) &
            (recommendations_df['Name'] != player_name)
        ]

        # 5. Sort the players by distance (lower is more similar) and return the top k
        top_k_recommendations = cluster_recommendations.sort_values(by='Distance').head(k)

        # Merge with original dataframe to show player attributes
        result = pd.merge(top_k_recommendations, self.df, on='Name')
        return result[['Name', 'Distance', 'OVR' if 'OVR' in self.df.columns else 'PAC', 
                       'Position' if 'Position' in self.df.columns else 'SHO']]

# --- Example Usage ---
if __name__ == "__main__":
    # NOTE: You need to download the 'FC 25 Players Data.csv' dataset first.
    # You can find datasets on platforms like Kaggle.
    # If the file is not found, a small dummy dataset will be used.
    DATA_FILE_PATH = '/kaggle/input/ea-sports-fc-25-database-ratings-and-stats/male_players.csv'

    try:
        # Initialize the recommender system
        recommender = PlayerRecommender(data_path=DATA_FILE_PATH, n_clusters=15)

        # --- Get recommendations for a specific player ---
        target_player = "Kevin De Bruyne"
        print(f"\n--- Finding recommendations for: {target_player} ---")
        recommendations = recommender.get_recommendations(target_player, k=20)

        if isinstance(recommendations, pd.DataFrame):
            print(f"Top 20 most similar players to {target_player}:")
            print(recommendations.to_string())
        else:
            print(recommendations)


        # --- Another example ---
        target_player_2 = "Pascal Struijk"
        print(f"\n--- Finding recommendations for: {target_player_2} ---")
        recommendations_2 = recommender.get_recommendations(target_player_2, k=20)

        if isinstance(recommendations_2, pd.DataFrame):
            print(f"Top 20 most similar players to {target_player_2}:")
            print(recommendations_2.to_string())
        else:
            print(recommendations_2)

    except (ValueError, FileNotFoundError) as e:
        print(f"\nAn error occurred: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
