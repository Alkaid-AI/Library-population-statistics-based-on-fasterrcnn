
with open('D:/AI_path/keras-frcnn-master/VOCdevkit/VOC2012/ImageSets/Main/aaaa_person_val.txt', 'r') as f:
        for line in f.readlines():
            # print(line[12:14])
           if line[12:14] == ' 1':
               newname=line.replace(line,line[0:11])
               newfile = open('D:/AI_path/keras-frcnn-master/VOCdevkit/VOC2012/ImageSets/Main/aaa.txt', 'a')
               newfile.write(newname + '\n')
               newfile.close()



