import pandas as pd
import json
from pymongo import MongoClient

# ── Connect to MongoDB ────────────────────────────────────────────────────
client = MongoClient('mongodb://localhost:27017/')
db = client['av_perception']

detections_col = db['detections']
performance_col = db['class_performance']
models_col = db['model_summary']

# ── Load detections ───────────────────────────────────────────────────────
print("Loading detections into MongoDB...")
detections_col.drop()
df = pd.read_csv('detections.csv')
records = df.to_dict('records')
detections_col.insert_many(records)
print(f"✅ Inserted {len(records)} detections")

# ── Load class performance ────────────────────────────────────────────────
print("Loading class performance...")
performance_col.drop()
perf_df = pd.read_csv('class_performance.csv')
perf_records = perf_df.to_dict('records')
performance_col.insert_many(perf_records)
print(f"✅ Inserted {len(perf_records)} class records")

# ── Load model summary ────────────────────────────────────────────────────
print("Loading model summary...")
models_col.drop()
with open('model_summary.json') as f:
    summary = json.load(f)
models_col.insert_one(summary)
print("✅ Inserted model summary")

# ── Verify ────────────────────────────────────────────────────────────────
print("\n📊 MONGODB VERIFICATION")
print("="*40)
print(f"Detections collection:   {detections_col.count_documents({})} docs")
print(f"Performance collection:  {performance_col.count_documents({})} docs")
print(f"Model summary:           {models_col.count_documents({})} doc")

print("\nSample detection:")
sample = detections_col.find_one({}, {'_id': 0})
for k, v in sample.items():
    print(f"  {k:20s}: {v}")

print("\nTop 5 classes by detection count:")
top = performance_col.find({}, {'_id': 0}).sort(
    'detection_count', -1).limit(5)
for doc in top:
    bar = '█' * int(doc['detection_count'] / 3)
    print(f"  {doc['class']:20s} {bar} {doc['detection_count']}")

client.close()
print("\n✅ MongoDB loading complete!")
