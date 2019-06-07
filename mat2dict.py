import scipy.io as io
import pickle

data = io.loadmat('Annotations.mat')
print('数据集加载完毕！')


def get_info(ann, is_train):  # is_train: bool型变量，表示该部分是否为训练集
    info = {
        'filename': ann[0][0],
        'folder': ann[1][0],
        'size': (ann[2][0][0][0][0][0], ann[2][0][0][1][0][0]),
        'database': ann[3][0][0][0][0]
    }

    if len(ann[3][0][0]) > 1:
        info['img_id'] = ann[3][0][0][1][0][0][0][0][0]
        info['ann_id'] = ann[3][0][0][1][0][0][1][0][0]

    people = ann[4][0]
    info['people'] = {}
    info['people']['num'] = len(people)
    for i in range(len(people)):
        person = people[i]
        info['people'][i] = {}
        info['people'][i]['body_box'] = [int(_) for _ in person[0][0]]

        if is_train:
            info['people'][i]['categories'] = [_[0] for _ in person[1][0][0][0][0]]
            info['people'][i]['continuous'] = [_[0][0] for _ in person[2][0][0]]
            '''
            info['people'][i]['valence'] = info['people'][i]['continuous'][0]
            info['people'][i]['arousal'] = info['people'][i]['continuous'][1]
            info['people'][i]['dominance'] = info['people'][i]['continuous'][2]
            '''

        else:
            info['people'][i]['categories'] = {}
            info['people'][i]['categories']['num'] = len(person[1][0])
            for j in range(len(person[1][0])):
                info['people'][i]['categories'][j] = [_[0] for _ in person[1][0][j][0][0]]

            if len(person[2]) > 0:
                info['people'][i]['combined_categories'] = [_[0] for _ in person[2][0]]
            else:
                info['people'][i]['combined_categories'] = []

            info['people'][i]['continuous'] = {}
            info['people'][i]['continuous']['num'] = len(person[3][0])
            for j in range(len(person[3][0])):
                info['people'][i]['continuous'][j] = [_[0][0] for _ in person[3][0][j]]

            info['people'][i]['combined_continuous'] = [_[0][0] for _ in person[4][0]]

        info['people'][i]['gender'] = person[3 if is_train else 5][0]
        info['people'][i]['age'] = person[4 if is_train else 6][0]

    return info


if __name__ == '__main__':
    annotations = {
        'train': {},
        'val': {},
        'test': {}
    }
    train = data['train'][0]
    val = data['val'][0]
    test = data['test'][0]

    for k in range(train.shape[0]):
        annotations['train'][k] = get_info(train[k], is_train=True)
    for k in range(val.shape[0]):
        annotations['val'][k] = get_info(val[k], is_train=False)
    for k in range(test.shape[0]):
        annotations['test'][k] = get_info(test[k], is_train=False)

    with open('Annotations.pkl', 'wb') as f:
        pickle.dump(annotations, f, pickle.HIGHEST_PROTOCOL)
