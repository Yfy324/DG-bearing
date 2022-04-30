from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

y_true = np.argmax(labels, axis=-1)
y_pred = np.argmax(predictions, axis=-1)
C = confusion_matrix(y_true, y_pred, labels=[0,1,2])
plt.matshow(C, cmap=plt.cm.Reds) # 根据最下面的图按自己需求更改颜色
# plt.colorbar()
for i in range(len(C)):
    for j in range(len(C)):
        plt.annotate(C[j, i], xy=(i, j), horizontalalignment='center', verticalalignment='center')
# plt.tick_params(labelsize=15) # 设置左边和上面的label类别如0,1,2,3,4的字体大小。
plt.ylabel('True label')
plt.xlabel('Predicted label')
# plt.ylabel('True label', fontdict={'family': 'Times New Roman', 'size': 20}) # 设置字体大小。
# plt.xlabel('Predicted label', fontdict={'family': 'Times New Roman', 'size': 20})
plt.show()
