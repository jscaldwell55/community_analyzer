import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans

class CommunityEngagementAnalyzer:
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.kmeans = None

    def prepare_sample_data(self):
        """Generate sample community engagement data"""
        n_users = 1000
        data = {
            'user_id': range(1, n_users + 1),
            'forum_posts': np.random.randint(0, 50, n_users),
            'discord_messages': np.random.randint(0, 200, n_users),
            'webinar_attendance': np.random.randint(0, 10, n_users),
            'office_hours_attendance': np.random.randint(0, 5, n_users),
            'documentation_views': np.random.randint(0, 300, n_users),
            'templates_used': np.random.randint(0, 20, n_users),
            'account_age_days': np.random.randint(1, 365, n_users)
        }
        return pd.DataFrame(data)

    def preprocess_data(self, data):
        """Preprocess the engagement data for clustering"""
        df = data.copy()
        
        # Select features for clustering
        features = ['forum_posts', 'discord_messages', 'webinar_attendance', 
                    'office_hours_attendance', 'documentation_views', 'templates_used', 'account_age_days']
        
        X = df[features]
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        return X_scaled, df

    def perform_clustering(self, X_scaled, n_clusters=5):
        """Perform K-means clustering on the preprocessed data"""
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = self.kmeans.fit_predict(X_scaled)
        return clusters

    def analyze_clusters(self, data, clusters):
        """Generate insights about each cluster"""
        df = data.copy()
        df['Cluster'] = clusters
        
        insights = []
        for cluster in range(len(np.unique(clusters))):
            cluster_data = df[df['Cluster'] == cluster]
            
            insight = {
                'cluster': cluster,
                'size': len(cluster_data),
                'avg_forum_posts': cluster_data['forum_posts'].mean(),
                'avg_discord_messages': cluster_data['discord_messages'].mean(),
                'avg_webinar_attendance': cluster_data['webinar_attendance'].mean(),
                'avg_office_hours': cluster_data['office_hours_attendance'].mean(),
                'avg_doc_views': cluster_data['documentation_views'].mean(),
                'avg_templates': cluster_data['templates_used'].mean(),
                'avg_account_age': cluster_data['account_age_days'].mean()
            }
            insights.append(insight)
            
        return pd.DataFrame(insights)
