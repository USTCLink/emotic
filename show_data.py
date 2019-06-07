import pickle
import cv2

path = 'D:/Datasets/emotic/emotic/'
with open('Annotations.pkl', 'rb') as f:
    annotations = pickle.load(f)

print('数据集加载完毕！')

print('数据集包含train，val，test三个部分，各部分中数据数量如下')

print('train:', len(annotations['train'].keys()))
print('val:', len(annotations['val'].keys()))
print('test:', len(annotations['test'].keys()))

if __name__ == '__main__':
    notice = '''
    本程序可以展示数据集中的某个数据
    请输入数据所属部分（train或val或test）及数据在该部分中的编号
    编号从0开始，注意不要超出范围
    示例：
        train 1926
        test 817
    每次展示会打开对应图片（对应人物用红色长方形标注），关闭图片后可输入下一个数据
    '''
    print(notice)

    while True:
        part, n = input('请输入：').split()
        n = int(n)
        assert part in ['train', 'val', 'test']
        assert 0 <= n < len(annotations[part].keys())
        train = part == 'train'
        info = annotations[part][n]

        filename = info['filename']
        print('filename:', filename)

        folder = info['folder']
        print('folder:', folder)
        img = cv2.imread(path + folder + '/' + filename)

        size = info['size']
        print('size:', size)

        database = info['database']
        print('database:', database)

        if 'img_id' in info and 'ann_id' in info:
            img_id = info['img_id']
            ann_id = info['ann_id']
            print('image_id:', img_id)
            print('annotation_id:', ann_id)

        people = info['people']
        for i in range(people['num']):
            print('person', i + 1, ':')
            person = people[i]
            body_box = person['body_box']
            left, top, right, bottom = body_box
            cv2.rectangle(img, (left, top), (right, bottom), (0, 0, 255))
            cv2.putText(img, str(i), (left, top + 30), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 0, 255), 1)

            categories = person['categories']
            print('categories:', categories)

            if not train:
                combined_categories = person['combined_categories']
                print('combined_categories:', combined_categories)

            continuous = person['continuous']
            print('continuous:', continuous)

            if not train:
                combined_continuous = person['combined_continuous']
                print('combined_continuous:', combined_continuous)

            gender = person['gender']
            print('gender:', gender)

            age = person['age']
            print('age:', age)

        cv2.imshow('Window', img)
        cv2.waitKey(0)
