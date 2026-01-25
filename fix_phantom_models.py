
import os
import sys

# List of specific phantom files to delete
files_to_delete = [
    r"C:\Users\Никола\.stocker\ml_metadata_mixed.json",
    r"C:\Users\Никола\.stocker\ml_metadata_mixed_bear.json",
    r"C:\Users\Никола\.stocker\ml_metadata_mixed_bull.json"
]

print("Starting cleanup of phantom models...")
for file_path in files_to_delete:
    if os.path.exists(file_path):
        try:
            os.remove(file_path)
            print(f"✅ Deleted: {file_path}")
        except Exception as e:
            print(f"❌ Error deleting {file_path}: {e}")
    else:
        print(f"ℹ️ File not found (already clean): {file_path}")

print("Cleanup finished. You can now retrain the Mixed strategy.")
