import cv2
import os
from detect_board import detect_board
from detect_shapes import detect_shapes

def main():
    dataset_path = r"E:\Collage\MIA\Task 11\Classical Detection\Data Set"
    results_path = r"E:\Collage\MIA\Task 11\Classical Detection\Results"
    os.makedirs(results_path, exist_ok=True)

    for filename in os.listdir(dataset_path):
        if filename.lower().endswith((".jpg", ".png", ".jpeg")):
            img_path = os.path.join(dataset_path, filename)
            img = cv2.imread(img_path)

            if img is None:
                print(f"❌ Skipping {filename} (could not load)")
                continue

            # Step 1: detect board
            board = detect_board(img)

            # Step 2: detect shapes inside board
            result = detect_shapes(board)

            # Step 3: save result
            save_path = os.path.join(results_path, f"result_{filename}")
            cv2.imwrite(save_path, result)
            print(f"✅ Processed {filename} → saved {save_path}")

if __name__ == "__main__":
    main()
