"""
Interactive Query Tester
Author: Afnan Khan (@Afnan-makstrix)
"""
import sys
import os
import cv2
from sift_landmark_system import LandmarkRecognitionPipeline

def main():
    query_file = sys.argv[1] if len(sys.argv) > 1 else "data/landmarks/eiffel_tower/ref_1.jpg"
    print(f"[Afnan Landmark Demo] Testing Query: {query_file}")
    
    pipe = LandmarkRecognitionPipeline()
    pipe.build_database("data/landmarks")

    img = cv2.imread(query_file)
    rec = pipe.recognize(img)
    print(rec['decision_message'])
    
    if rec['is_recognized']:
        top = rec['top_score']
        loc = pipe.localize(img, rec['query_kps'], top['ref']['bgr'], top['ref']['kps'], top['matches'])
        cv2.imwrite("results/query_localized.jpg", loc['image'])
        print("-> Saved localized output to results/query_localized.jpg")

if __name__ == '__main__':
    main()
