# 🎮 EA Sports FC 25 Player Recommender System
This project is a Player Recommendation System for EA Sports FC 25. It uses machine learning techniques to find players with similar play styles and attributes, making it useful for team building, scouting alternatives, or analyzing player similarities.

# 🚀 Features
💡 Similarity Recommendation: Find players most similar to a given player based on their attributes using Euclidean distance.

📊 Attribute-Based Clustering:
Groups players into clusters using K-Means Clustering, helping you explore groups of similar types (e.g., playmakers, wingers, defenders).

🧼 Data Preprocessing:
Handles missing values, filters out goalkeepers, and standardizes numerical attributes for fair comparison.

🔍 Fuzzy Matching:
If a player name is mistyped, the system will suggest the closest matching player using intelligent text matching.

🧪 Dummy Dataset Fallback:
If the real dataset isn't found, a small dummy dataset with famous players (e.g., Messi, Ronaldo) is used for demonstration purposes.

# 📂 Dataset
The recommender expects a dataset in CSV format containing player attributes.
Recommended dataset: male_players.csv from the EA Sports FC 25 database (Kaggle or similar sources).

Essential attributes (used when available):

Physical (e.g., Strength, Stamina)

Technical (e.g., Dribbling, Passing)

Mental (e.g., Composure, Vision)

Defensive/Offensive stats

Fallback attributes (for small or dummy datasets): OVR, PAC, SHO, PAS, DRI, DEF, PHY.

# 🧠 Technologies Used
- Python
- Pandas for data manipulation
- Scikit-learn for StandardScaler and KMeans clustering
- Scipy for distance calculations
- difflib for fuzzy string matching

# 🛠️ How It Works
Data Loading & Preprocessing:
Reads the player data, filters relevant attributes, handles missing values, and normalizes the data.

Similarity Calculation:
Computes a Euclidean distance matrix among all players after scaling attributes.

Clustering:
Applies K-Means clustering to group players with similar profiles.

Recommendation:
For a given player, finds the k most similar players within the same cluster, ordered by similarity.
