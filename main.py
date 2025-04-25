import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
# Load data
transactions = pd.read_csv("data/transactions_large.csv")
subscriptions = pd.read_csv("data/subscriptions_large.csv")
products = pd.read_csv("data/products_large.csv")

purchase_freq = transactions.groupby("customer_id")["transaction_date"].count().rename("purchase_count")


aov = transactions.groupby("customer_id")["price_paid"].mean().rename("avg_order_value")


trans_merged = transactions.merge(products, on="product_id")
category_affinity = (
    trans_merged.groupby(["customer_id", "category"])
    .size()
    .unstack(fill_value=0)
    .apply(lambda x: x / x.sum(), axis=1)
)


features = pd.concat([purchase_freq, aov, category_affinity], axis=1).fillna(0)


active_subs = subscriptions[subscriptions["subscription_status"] == "active"]
subscribed_customers = set(active_subs["customer_id"])
all_customers = set(features.index)

non_subscribed_customers = list(all_customers - subscribed_customers)


scaler = StandardScaler()
features_scaled = pd.DataFrame(
    scaler.fit_transform(features),
    index=features.index,
    columns=features.columns
)

def recommend_for_non_subscriber(customer_id, top_n_users=3, top_n_products=5):
    if customer_id not in non_subscribed_customers:
        print(f"{customer_id} is not a valid non-subscriber.")
        return []
    
    
    target_vec = features_scaled.loc[customer_id].values.reshape(1, -1)
    
    
    subscribed_features = features_scaled.loc[list(subscribed_customers)]
    
    
    sims = cosine_similarity(target_vec, subscribed_features)[0]
    
    
    sim_df = pd.DataFrame({
        "customer_id": subscribed_features.index,
        "similarity": sims
    }).sort_values(by="similarity", ascending=False).head(top_n_users)
    
    similar_users = sim_df["customer_id"].tolist()
    
    print(f"\nTop {top_n_users} similar users to {customer_id}:\n")
    print(sim_df.to_string(index=False))

    
    print("\nFeature summary of similar users:")
    print(features.loc[similar_users])

    
    similar_subs = active_subs[active_subs["customer_id"].isin(similar_users)]
    candidate_products = set(similar_subs["product_id"])
    
    
    already_purchased = set(transactions[transactions["customer_id"] == customer_id]["product_id"])
    
    
    recommendations = list(candidate_products - already_purchased)
    
    
    popular_products = (
        similar_subs[similar_subs["product_id"].isin(recommendations)]
        .groupby("product_id").size().sort_values(ascending=False)
        .head(top_n_products).index.tolist()
    )
    
    return popular_products

cust_id = "CUST_0017"
rec = recommend_for_non_subscriber(cust_id)
print(f"\n Final Recommendations for {cust_id}: {rec}")