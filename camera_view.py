import os
import cv2
from PIL import Image
import imagehash
import pandas as pd
from utils.log_utils import logger
# from config import settings

def compare_hashes(hash1, hash2):
    """Compare two image hashes and return the similarity score."""
    return hash1 - hash2

def compare_consecutive_frames(video_path, csv_folder_path, threshold=20, debug_path=None):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        logger.error("Error: Could not open video.")
        return

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    results = []
    prev_hash = None
    for i in range(frame_count - 1):
        ret, frame = cap.read()
        if not ret:
            print(f"Error: Could not read frame {i}")
            logger.error(f"Error: Could not read frame {i}")
            break
        
        current_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        current_hash = imagehash.phash(current_frame)
        
        if prev_hash is not None:
            similarity = compare_hashes(prev_hash, current_hash)
            if similarity >= threshold:
            #     print(f"Comparing frames {i-1} and {i}: Similarity score = {similarity} -> Similar")
            # else:
                print(f"Comparing frames {i-1} and {i}: Similarity score = {similarity} -> Not Similar")
                logger.info(f"Comparing frames {i-1} and {i}: Similarity score = {similarity} -> Not Similar")
        
            results.append({
                'Frame Pair': f"{i-1}-{i}",
                'Similarity Score': similarity
            })
        
        prev_hash = current_hash

    cap.release()

    # Create a DataFrame from the results
    df = pd.DataFrame(results)

    # Save DataFrame to CSV
    # csv_folder_path = os.path.join(settings.DEBUG_PATH, "camera_view")
    csv_folder_path = os.path.join(debug_path, "camera_view")
    if not os.path.exists(csv_folder_path):
        os.makedirs(csv_folder_path)
    csv_filename = os.path.join(csv_folder_path, 'frame_similarity_scores.csv')
    df.to_csv(csv_filename, index=False)
    print(f"Similarity scores saved to {csv_filename}")
    logger.info(f"Similarity scores saved to {csv_filename}")
    return csv_filename, frame_count


def process_similarity_scores(filepath, output_csv_path, threshold=20):
    results = []
    df = pd.read_csv(filepath)
    similarity_scores = df['Similarity Score'].tolist()
    frame_pairs = df['Frame Pair'].tolist()
    
    results.append({
        'File': filepath.split("/")[-1],
        'Frame Pair': '------',
        'left_avg': '-----',
        'Similarity Score': '-------',
        'right_avg': '-------'
    })
    
    for idx, similarity in enumerate(similarity_scores):
        if similarity >= threshold:
            # Collect previous 3 non-zero scores
            prev_scores = []
            i = idx - 1
            while i >= 0 and len(prev_scores) < 3:
                if similarity_scores[i] > 0:
                    prev_scores.append(similarity_scores[i])
                i -= 1
            
            # Collect next 3 non-zero scores
            next_scores = []
            i = idx + 1
            while i < len(similarity_scores) and len(next_scores) < 3:
                if similarity_scores[i] > 0:
                    next_scores.append(similarity_scores[i])
                i += 1
            
            if not prev_scores or not next_scores:
                continue
            
            avg_prev = sum(prev_scores) / len(prev_scores)
            avg_next = sum(next_scores) / len(next_scores)
                
            if avg_prev >= similarity / 3 and avg_next >= similarity / 3:
                continue 
            
            results.append({
                # 'File': file,
                'Frame Pair': frame_pairs[idx],
                'left_avg': avg_prev,
                'Similarity Score': similarity,
                'right_avg': avg_next
            })
    
    # Save results to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_csv_path, index=False)
    print(f"Results saved to {output_csv_path}")
    logger.info(f"Results saved to {output_csv_path}")

def drop_duplicate_frames(output_csv_path= None, video_path= None):
    if output_csv_path is None:
        output_csv_path = "/Users/vaibhav/Desktop/football_projects/fb_model_engine/football_analysis/camera_movement_estimator/data/output_camera_switch/result2.csv"
    df_old = pd.read_csv(output_csv_path)
    df_new = pd.read_csv(output_csv_path)
    df_new.drop(columns= ['File', 'left_avg', 'right_avg', 'Similarity Score'], inplace=True)
    df_new = df_new.iloc[1:].reset_index(drop=True)
    # Open the video file
    video = cv2.VideoCapture(video_path)
    
    # Extract frame numbers from the 'Frame Pair' column
    start_numbers = []
    end_numbers = []
    
    for pair in df_new['Frame Pair']:
        start, end = map(int, pair.split('-'))
        start_numbers.append(start)
        end_numbers.append(end)
    
    # Initialize variables
    frame_count = 0
    hsv_prev = None
    rows_to_drop = []
    # Read frames from the video
    while True:
        ret, frame = video.read()
        
        if not ret:
            print(f"Failed to read frame {frame_count}")
            logger.error(f"Failed to read frame {frame_count}")
            break
        
        # If the current frame matches one of the start numbers
        if frame_count in start_numbers and hsv_prev is None:
            # Convert the frame to HSV color space
            hsv_prev = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            hist1 = cv2.calcHist([hsv_prev], [0, 1, 2], None, [50, 50, 50], [0, 256, 0, 256, 0, 256])
            hist1 = cv2.normalize(hist1, hist1)

        # If the current frame matches one of the end numbers and there's a previous frame to compare
        elif frame_count in end_numbers and hsv_prev is not None:
            hsv_curr = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            hist2 = cv2.calcHist([hsv_curr], [0, 1, 2], None, [50, 50, 50], [0, 256, 0, 256, 0, 256])
            hist2 = cv2.normalize(hist2, hist2)
            
            # Compare the histograms using correlation
            similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
            if frame_count in start_numbers:
                hsv_prev=hsv_curr
                hist1=hist2
            else:    
                hsv_prev = None  # Reset after comparison // if current frame count in the start numbers hsv_prev=hsv_curr 
            
            # Find the corresponding index in the DataFrame where this pair belongs
            idx = df_new[(df_new['Frame Pair'] == f"{start_numbers[0]}-{frame_count}")].index
            if len(idx) > 0 and (similarity > 0.90):
                rows_to_drop.append(idx[0])
                # df_new.at[idx[0], 'Hist_score'] = similarity
                # print(f"Histogram similarity for frame pair {start_numbers[0]}-{frame_count}: {similarity}")
            
            # Remove processed pair
            start_numbers.pop(0)
            end_numbers.pop(0)

        # Increment the frame count
        frame_count += 1
        
        # Exit the loop when all frames have been processed
        if frame_count > max(end_numbers, default=0):
            break
    
    # Release the video object
    video.release()
    # Drop rows with similarity > 0.90
    df_new = df_new.drop(rows_to_drop)
    df = df_old.merge(df_new, how = 'right', on= 'Frame Pair')
    df = pd.concat([df_old.iloc[[0]], df], ignore_index=True)
    df.to_csv(output_csv_path, index=False)
    # Reset index after dropping rows
    # df_new = df_new.reset_index(drop=True)
    print(f"Drop dulicates done and saved to: {output_csv_path}")
    logger.info(f"Drop duplicates done and saved to: {output_csv_path}")


def frame_change_compatibility(output_csv_path=None, frame_count=1098):
    if output_csv_path is None:
        output_csv_path = "/Users/vaibhav/Desktop/football_projects/fb_model_engine/football_analysis/camera_movement_estimator/data/output_camera_switch/result2.csv"
        
    df = pd.read_csv(output_csv_path)
    frame_pair_list = df['Frame Pair'][1:].to_list()
    frame_pair_list = [0]+[   int(val.split("-")[1]) for val in frame_pair_list]+[frame_count+1]
    # print(frame_pair_list)
    # frame_pair_list  = ["0-"+frame_pair_list[0].split("-")[0]] + frame_pair_list + [frame_pair_list[-1].split("-")[-1]+"-"+str(frame_count+1)]
    frame_change_count = 0
    new_df  = pd.DataFrame.from_dict({"frame_nun":[i for i in range(frame_count+1)]})
    new_df['frame_change'] = [0]*(frame_count+1)
    # print(new_df)
    
    for first_idx, second_idx in zip(frame_pair_list, frame_pair_list[1:]):
        # first_idx, second_idx = map(int, frame_chng.split("-"))
        new_df.loc[first_idx: second_idx-1, 'frame_change'] = [frame_change_count] * (second_idx - first_idx)
        # print(frame_change_count, first_idx, second_idx)
        frame_change_count = frame_change_count + 1
        

    new_df.to_csv(output_csv_path.split(".")[0]+"_frame_change.csv")
    return output_csv_path.split(".")[0]+"_frame_change.csv"



def camera_view(video_path = '/Users/vaibhav/Desktop/football_projects/fb_model_engine/football_analysis/data/input_video_dev_highlights/-fc-bayern-münchen-vs-bayer-04-leverkusen-34.mp4', csv_folder_path='/Users/vaibhav/Desktop/football_projects/fb_model_engine/football_analysis/camera_movement_estimator/data/output_camera_switch',debug_path=None):
    
    # Example usage
    
    # csv_folder_path = '/Users/vaibhav/Desktop/football_projects/fb_model_engine/football_analysis/camera_movement_estimator/data/output_camera_switch'
    csv_filename, frame_count = compare_consecutive_frames(video_path, csv_folder_path,debug_path=debug_path)
    # Example usage
    output_csv_path = os.path.join(csv_folder_path, "result2.csv")
    # output_csv_path = '/Users/vaibhav/Desktop/football_projects/fb_model_engine/football_analysis/camera_movement_estimator/data/result2.csv'
    
    process_similarity_scores(csv_filename, output_csv_path)
    drop_duplicate_frames(output_csv_path, video_path)
    frame_change_csv_path = frame_change_compatibility(output_csv_path, frame_count)
    return frame_change_csv_path

# def main():
#     # video_path = "/home/ubuntu/development/fb_model_engine/football_analysis/data/input_video/small_test.mp4"
#     csv_folder_path = "../data/debug/camera_view"

#     video_path =   "../data/input_video/segment.mp4"
#     path = camera_view(video_path=video_path,csv_folder_path=csv_folder_path,debug_path="../data/debug")
#     print(path)
#     # frame_change_compatibility()

# if __name__=='__main__':
#     main()