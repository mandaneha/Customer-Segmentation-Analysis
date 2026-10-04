import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

df = pd.read_csv("dataset/customer_data.csv")

print("First 5 Rows:")
print(df.head())
print("\nDataset Shape:")
print(df.shape)
print("\nColumn Names:")
print(df.columns.tolist())

print("\nDataset Information:")
df.info()

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:")
print(df.duplicated().sum())

print("\nDescriptive Statistics:")
print(df.describe())

df = df.drop_duplicates()

numeric_columns = [
    "Age", "AnnualIncome", "PurchaseFrequency", "AverageOrderValue",
    "TotalSpend", "RecencyDays", "CategoriesPurchased",
    "DiscountUsagePct", "CustomerTenureMonths", "SatisfactionScore"
]

df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors="coerce")
df = df.dropna(subset=numeric_columns)

df["ChurnFlag"] = pd.to_numeric(df["ChurnFlag"], errors="coerce")
df = df.dropna(subset=["ChurnFlag"])

print("\nDataset After Cleaning:")
print(df.shape)

print("\nGender Distribution:")
print(df["Gender"].value_counts())

print("\nRegion Distribution:")
print(df["Region"].value_counts())

print("\nPreferred Category Distribution:")
print(df["PreferredCategory"].value_counts())

features = [
    "AnnualIncome",
    "PurchaseFrequency",
    "AverageOrderValue",
    "TotalSpend",
    "RecencyDays"
]

X = df[features]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

inertia = []

for k in range(2, 11):
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    model.fit(X_scaled)
    inertia.append(model.inertia_)

print("\nElbow Method Results:")

for k, value in zip(range(2, 11), inertia):
    print(f"K = {k}: Inertia = {value:.2f}")

plt.figure(figsize=(8, 5))
plt.plot(range(2, 11), inertia, marker="o")
plt.title("Elbow Method for Optimal Number of Clusters")
plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")
plt.xticks(range(2, 11))
plt.tight_layout()
plt.savefig("elbow_method.png", dpi=300)
plt.close()

model = KMeans(n_clusters=4, random_state=42, n_init=10)
df["Cluster"] = model.fit_predict(X_scaled)

print("\nCluster Distribution:")
print(df["Cluster"].value_counts().sort_index())

cluster_summary = df.groupby("Cluster")[features].mean().round(2)

print("\nCluster Summary:")
print(cluster_summary)

cluster_summary.to_csv("cluster_summary.csv")

segment_names = {
    0: "High-Value Customers",
    1: "At-Risk Customers",
    2: "Regular Customers",
    3: "Valuable Active Customers"
}

df["CustomerSegment"] = df["Cluster"].map(segment_names)

segment_analysis = df.groupby("CustomerSegment").agg(
    CustomerCount=("CustomerID", "count"),
    AverageAge=("Age", "mean"),
    AverageIncome=("AnnualIncome", "mean"),
    AveragePurchaseFrequency=("PurchaseFrequency", "mean"),
    AverageOrderValue=("AverageOrderValue", "mean"),
    AverageTotalSpend=("TotalSpend", "mean"),
    AverageRecency=("RecencyDays", "mean"),
    AverageCategoriesPurchased=("CategoriesPurchased", "mean"),
    AverageDiscountUsage=("DiscountUsagePct", "mean"),
    AverageTenure=("CustomerTenureMonths", "mean"),
    AverageSatisfaction=("SatisfactionScore", "mean"),
    ChurnRate=("ChurnFlag", "mean")
).round(2)

segment_analysis["ChurnRate"] = (
    segment_analysis["ChurnRate"] * 100
).round(2)

print("\nCustomer Segment Analysis:")
print(segment_analysis)

segment_analysis.to_csv("segment_analysis.csv")
df.to_csv("customer_segmentation_final.csv", index=False)

print("\nCustomer Segmentation Completed")
print("\nFiles Created:")
print("1. elbow_method.png")
print("2. cluster_summary.csv")
print("3. segment_analysis.csv")
print("4. customer_segmentation_final.csv")

print("\nFinal Dataset Shape:")
print(df.shape)

print("\nCustomer Segments:")
print(df["CustomerSegment"].value_counts())

pd.set_option("display.max_columns", None)

print("\nFULL SEGMENT ANALYSIS:")
print(segment_analysis)

print("\nFULL CLUSTER SUMMARY:")
print(cluster_summary)