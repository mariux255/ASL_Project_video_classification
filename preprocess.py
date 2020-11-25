import csv 
import os
import json
import cv2

def main(frame_location="/Users/mjo/Desktop/WLASL/Processed_data/",json_file_path="/Users/mjo/Desktop/WLASL/WLASL_v0.3.json",extract_frames_bool = True,video_file_path="/Users/mjo/Desktop/WLASL/WLASL2000"):
    # Defining count_dictionary which contains the number of videos for each class
    # and defining video_id_dictionary which has all of the video id's for each class.
    with open(json_file_path, "r") as read_file:
        data = json.load(read_file)

    count_dictionary = {}
    video_id_dictionary = {}

    for instance in data:
        current_label = instance['gloss']
        count_dictionary[current_label] = 0
        video_id_dictionary[current_label] = []
        inner_array = instance['instances']
        for video in inner_array:
            count_dictionary[current_label] += 1
            video_id_dictionary[current_label].append(video['video_id'])

    # Creating a list for 100, 200, 500, 1000 and 2000 classes with the highest amount of videos.
    count = 0
    labels_100 = []
    labels_200 = []
    labels_500 = []
    labels_1000 = []
    labels_2000 = []

    for key in count_dictionary:
        if count < 100:
            labels_100.append(key)
        if count < 200:
            labels_200.append(key)
        if count < 500:
            labels_500.append(key)
        if count < 1000:
            labels_1000.append(key)

        labels_2000.append(key)

        count += 1

    # Creating folders where each video has a folder with all of its frames inside.

    if extract_frames_bool == True:
        #print(len(self.labels_100))
        exctract_frames(labels_100,frame_location, video_file_path,video_id_dictionary)
        #print(len(self.labels_200))
        exctract_frames(labels_200,frame_location, video_file_path,video_id_dictionary)
        #print(len(self.labels_500))
        exctract_frames(labels_500,frame_location, video_file_path,video_id_dictionary)
        #print(len(self.labels_1000))
        exctract_frames(labels_1000,frame_location, video_file_path, video_id_dictionary)
        #print(len(self.labels_2000))
        exctract_frames(labels_2000,frame_location, video_file_path, video_id_dictionary)
    return count_dictionary, video_id_dictionary, labels_100, labels_200, labels_500, labels_1000, labels_2000
def exctract_frames(labels_x,frame_location, video_file_path,video_id_dictionary):

    video_count = 0
    num_classes = len(labels_x)

    if not os.path.exists("{}".format(frame_location)):
        os.mkdir("{}".format(frame_location))

    if not os.path.exists("{}/{}".format(frame_location, num_classes)):
        os.mkdir("{}/{}".format(frame_location, num_classes))

    for label in labels_x:
        for video in video_id_dictionary[label]:
            video_capture = cv2.VideoCapture(
                os.path.join(video_file_path, video + ".mp4"))
            success, image = video_capture.read()
            count = 0

            if video_count % 10 == 0:
                print(video_count)

            if not os.path.exists("{}/{}/{}".format(frame_location, num_classes, video)):
                os.mkdir("{}/{}/{}".format(frame_location, num_classes, video))
            while success:
                cv2.imwrite("{}/{}/{}/{}".format(frame_location, num_classes,
                                                video, "frame%d.jpg" % count), image)
                success, image = video_capture.read()
                count += 1
            video_count += 1
def WriteToCsv(filename, header, rows):
    with open(filename, 'w') as csvfile:
        csvwriter = csv.writer(csvfile)

        csvwriter.writerow(header1)
        csvwriter.writerows(rows)

def CountVideos(labels_x, video_id_dictionary):
    total = 0
    for label in labels_x:
        video_list = video[label]
        total += len(video_list)
    return total

count, video, l100, l200, l500, l1000, l2000 = main()
categories = [l100, l200, l500, l1000, l2000]
header1 = ['Class', 'Count']
header2 = ['Classes', 'Videos']
rows1 = []
rows2 = []
filename1 = "LabelsVsCount.csv"
filename2 = "CategoryVsCount.csv"
for label in count:
    rows1.append(['{}'.format(label),'{}'.format(count[label])])
WriteToCsv(filename1, header1, rows1)
for category in categories:
    rows2.append(['{}'.format(str(len(category))), '{}'.format(CountVideos(category, video))])

WriteToCsv(filename2,header2,rows2)
print("Preprocess finished")