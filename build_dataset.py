import os
import cv2
import numpy as np
import glob
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# --- 1. DEFINE LOCAL PATHS ---
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_DIR, 'data')
EXTRACTED_DATA_PATH = os.path.join(DATA_DIR, 'C-NMC_Leukemia')
PROCESSED_DATA_FILE = os.path.join(DATA_DIR, 'leukemia_data_processed.npz')

IMG_SIZE = 128

# --- 2. CHECK FOR DATA ---
if not os.path.exists(EXTRACTED_DATA_PATH):
    print("❌ FATAL ERROR: Dataset folder not found!")
else:
    print("✅ Dataset found locally. Starting processing...")

    # --- 3. PROCESS THE IMAGES (FINAL CORRECTED VERSION) ---
    if os.path.exists(PROCESSED_DATA_FILE):
        print("✅ Processed data file already exists. Nothing to do.")
    else:
        # === THIS IS THE CORRECTED PATH ===
        # The script now looks for the extra nested 'fold_*' directory to match your exact structure.
        print("Searching for images in the nested 'fold_*/fold_*' structure...")
        all_leukemia_paths = glob.glob(os.path.join(EXTRACTED_DATA_PATH, 'fold_*', 'fold_*', 'all', '*.bmp'))
        all_healthy_paths = glob.glob(os.path.join(EXTRACTED_DATA_PATH, 'fold_*', 'fold_*', 'hem', '*.bmp'))
        # =================================

        all_image_paths = all_leukemia_paths + all_healthy_paths
        all_labels = [1] * len(all_leukemia_paths) + [0] * len(all_healthy_paths)

        if not all_image_paths:
            print(
                "❌ ERROR: Still found 0 images. The folder structure is incorrect. Please re-check the 'tree data' output.")
        else:
            images, labels = [], []
            for i, image_path in enumerate(tqdm(all_image_paths, desc="Processing Images")):
                image = cv2.imread(image_path)
                if image is not None:
                    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
                    image = image / 255.0
                    images.append(image)
                    labels.append(all_labels[i])

            X = np.array(images)
            y = np.array(labels)

            print(f"\n✅ Image processing complete. Found {len(X)} images.")

            # --- 4. SPLIT THE DATA ---
            print("\n⚖️ Splitting data...")
            X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42,
                                                                        stratify=y)
            X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42,
                                                              stratify=y_train_val)
            print(f"Training: {len(X_train)}, Validation: {len(X_val)}, Test: {len(X_test)}")

            # --- 5. SAVE THE FINAL FILE ---
            print(f"\n💾 Saving processed data...")
            np.savez_compressed(PROCESSED_DATA_FILE,
                                X_train=X_train, y_train=y_train,
                                X_val=X_val, y_val=y_val,
                                X_test=X_test, y_test=y_test)
            print("\n🎉 All done! Your dataset is now ready for model training.")