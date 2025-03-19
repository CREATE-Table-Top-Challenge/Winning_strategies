import os
import cv2
import pandas as pd
import json

def draw_boxes_and_text(image, boxes, text):
    for box in boxes:
        cv2.rectangle(image, (box['xmin'], box['ymin']), (box['xmax'], box['ymax']), (0, 255, 0), 2)
        cv2.putText(image, box['class'], (box['xmin'], box['ymin']-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
    cv2.putText(image, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
    return image
def fix_json_format(json_str):
    return json_str.replace("'", '"')

directory = 'Training_Data_Part1'
for folder in os.listdir(directory):
    csv_file = os.path.join(directory, folder, f'{folder}_Labels.csv')
    df = pd.read_csv(csv_file)
    df['Tool bounding box'] = df['Tool bounding box'].apply(fix_json_format).apply(json.loads)
    for index, row in df.iterrows():
        image_file = os.path.join(directory, folder, row['FileName'])
        image = cv2.imread(image_file)
        image = draw_boxes_and_text(image, row['Tool bounding box'], row['Overall Task'])
        image = cv2.resize(image, (1600, 900))
        cv2.imshow(folder, image)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('p'):  # Check if 'p' is pressed
            cv2.waitKey(-1)  # wait until any key is pressed
        elif key == ord('n'):  # Check if 'n' is pressed
            cv2.destroyAllWindows()
            break  # play next video
        elif key == ord('q'):  # Check if 'q' is pressed
            cv2.destroyAllWindows()
            exit(0)
    cv2.waitKey(-1)  # pause after all frames in the current folder have been displayed
    cv2.destroyAllWindows()


cv2.destroyAllWindows()
