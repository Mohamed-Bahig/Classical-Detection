import cv2
import os
from utilities.Board_Detection import detect_board
from utilities.Shaps_Detection import detect_shapes

def main():
    dataset_path = r"E:\Collage\MIA\Task 11\Classical Detection\Data Set"
    results_path = r"E:\Collage\MIA\Task 11\Classical Detection\Results"
    os.makedirs(results_path, exist_ok=True)

    for filename in os.listdir(dataset_path):
        if filename.lower().endswith((".jpg", ".png", ".jpeg")):
            img_path = os.path.join(dataset_path, filename)
            img = cv2.imread(img_path)

            if img is None:
                print(f"[❌] Skipping {filename} (could not load)")
                continue

            # Step 1: detect board
            board = detect_board(img)
            if board is None:
                print(f"[⚠️] Skipping {filename} (board not detected)")
                continue

            # Step 2: detect shapes inside board
            result = detect_shapes(board)
            if result is None:
                print(f"[⚠️] Skipping {filename} (shape detection failed)")
                continue

            # Step 3: save result
            save_name = f"result_{filename}"
            save_path = os.path.join(results_path, save_name)
            cv2.imwrite(save_path, result)

            print(f"[✅] Processed {filename} → saved as {save_name}")

if __name__ == "__main__":
    main()
