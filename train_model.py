import pandas as pd
import re
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

print("--- STARTING TRAINING SESSION ---")

# 1. Load Data
try:
    df = pd.read_csv('final_master_dataset.csv')
    print(f"✅ Data Loaded. Total Articles: {len(df)}")
except FileNotFoundError:
    print("❌ Error: 'final_master_dataset.csv' not found. Please run merge_data.py first.")
    exit()

# ---------------------------------------------------------
# 🛠️ CRITICAL DATA CLEANING BLOCK (Do not remove)
# ---------------------------------------------------------
print("--- CLEANING DATA ---")

# 1. Standardize column names
df.columns = df.columns.str.strip()

# 2. Map Text Labels to Numbers
# This prevents the "True"/"Fake" text labels from being deleted as bad data
label_mapping = {
    'True': 1, 'TRUE': 1, 'true': 1, 'Real': 1, 'REAL': 1, '1': 1, 1: 1,
    'Fake': 0, 'FAKE': 0, 'fake': 0, '0': 0, 0: 0
}
df['label'] = df['label'].map(label_mapping)

# 3. Drop unmappable garbage rows
initial_count = len(df)
df.dropna(subset=['label', 'text'], inplace=True)
dropped_count = initial_count - len(df)

# 4. Ensure labels are integers
df['label'] = df['label'].astype(int)

print(f"Removed {dropped_count} unreadable rows.")
print(f"Final Training Count: {len(df)}") 
print("-----------------------")
# ---------------------------------------------------------

# 2. CLEANING FUNCTION
def wordopt(text):
    text = str(text).lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d', '', text)
    text = re.sub(r'\n', ' ', text)
    return text

print("Cleaning text data (this might take a moment)...")
df['text'] = df['text'].apply(wordopt)

# 3. SPLIT
x = df['text']
y = df['label']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# 4. VECTORIZE
print("Vectorizing text...")
vectorization = TfidfVectorizer()
xv_train = vectorization.fit_transform(x_train)
xv_test = vectorization.transform(x_test)

# 5. TRAIN
print("Training Model (LinearSVC)...")
model = LinearSVC()
model.fit(xv_train, y_train)

# 6. EVALUATE
print("Calculating Accuracy...")
pred = model.predict(xv_test)

acc = accuracy_score(y_test, pred)
precision, recall, f1, _ = precision_recall_fscore_support(y_test, pred, average='weighted')

print(f"✅ Accuracy:  {round(acc*100, 2)}%")
print(f"✅ Precision: {round(precision*100, 2)}%")
print(f"✅ Recall:    {round(recall*100, 2)}%")
print(f"✅ F1 Score:  {round(f1*100, 2)}%")

# 7. GENERATE GRAPHS

# Confusion Matrix
plt.figure(figsize=(6, 5))
cm = confusion_matrix(y_test, pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Fake', 'Real'], yticklabels=['Fake', 'Real'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig('confusion_matrix.png')
print("📊 Saved 'confusion_matrix.png'")

# Bar Chart
plt.figure(figsize=(8, 6))
metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
values = [acc, precision, recall, f1]
colors = ['#4CAF50', '#2196F3', '#FF9800', '#f44336']

bars = plt.bar(metrics, values, color=colors, edgecolor='black')
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, height, 
             f'{round(height*100, 1)}%', 
             ha='center', va='bottom', fontweight='bold')

plt.ylim(0, 1.1)
plt.title('Model Performance Metrics')
plt.ylabel('Score (0-1)')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig('bar_chart.png')
print("📊 Saved 'bar_chart.png'")

# 8. SAVE
joblib.dump(model, 'model.pkl')
joblib.dump(vectorization, 'vectorizer.pkl')
print("🎉 Model Saved Successfully!")