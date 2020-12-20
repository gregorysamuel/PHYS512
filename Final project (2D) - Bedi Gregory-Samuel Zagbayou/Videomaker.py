#Cleared from the names of my directories, etc
import imageio
import os

fileList = []
i=0
#Folder path
dir=
#Name without the index
file_begin=
#Video name
vid_name=
frame_rate=
for file in os.listdir(r+dir):
    if file.startswith(r+file_begin):
        i=i+1
        
for j in range(i):
    complete_path = r+dir+file_begin+str(j)+'.png' 
    fileList.append(complete_path)

writer = imageio.get_writer(vid_name, fps=frame_rate)

for im in fileList:
    writer.append_data(imageio.imread(im))
writer.close()