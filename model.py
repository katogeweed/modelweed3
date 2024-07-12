#データクレンジング
#画像を読み込み、かさ増しする作業

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense, Dropout, Flatten, Input
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras import optimizers
from time import sleep
from sklearn.metrics import f1_score

#雑草画像を格納するリストを作成
img_birodomozuika=[]
img_himeodorikosou=[]
img_yagurumagiku=[]

img_list=[img_birodomozuika,img_himeodorikosou,img_yagurumagiku]

weed_list=["birodomozuika",
           "himeodorikosou",
           "yagurumagiku"]

# 画像のサイズを指定
image_size=224

for index,weed in enumerate(weed_list):
    dir_name=os.path.join(r"C:\Users\admin\Desktop\VGG\weed_dataset",weed)
    path = os.listdir(dir_name)
    for i in range(len(path)):
        img_path = os.path.join(dir_name, path[i])
        img = cv2.imread(img_path)
        if img is None:
            print(f"Error: Unable to read image at {img_path}")
            continue
        img=cv2.resize(img,(image_size,image_size))
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        img_list[index].append(img)
        
        # img=cv2.flip(img,0)#上下反転
        # img_list[index].append(img)
        
        # img=cv2.flip(img,1)#左右反転
        # img_list[index].append(img)
        
        # img=cv2.flip(img,0)#上下反転
        # img_list[index].append(img)
 
#各雑草がカテゴリごとに格納できている確認       
for index, weed in enumerate(weed_list):
    print(f"{weed} images: {len(img_list[index])}")

#画像が反転しているかの確認    
#for x in img_birodomozuika:
    #plt.imshow(x)
    #plt.show()









#データ学習
#np.arrayでXに画像学習、yに正解ラベルを代入
X = np.array(img_birodomozuika + img_himeodorikosou + img_yagurumagiku)
y = np.array([0]*len(img_birodomozuika) + [1]*len(img_himeodorikosou) + [2]*len(img_yagurumagiku))

# 配列のラベルをシャッフルする
# Xにの長さに基づいて連番の配列を生成する。
# その連番の配列をランダムに並び替えたインデックス配列rand_indexを生成する。
# rand_indexを使って、xとyを同じ順序でランダムに並び替える。
rand_index = np.random.permutation(np.arange(len(X)))
X=X[rand_index]
y=y[rand_index]

#学習データと検証データを用意
X_train = X[:int(len(X)*0.8)]
y_train = y[:int(len(y)*0.8)]
X_test = X[int(len(X)*0.8):]
y_test = y[int(len(y)*0.8):]
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


# 正解ラベルをone-hotベクトルの形にする
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Karasモデルに入力する画像データの形状を指定し、入力層定義している。
# この入力層を使って、後続の層を構築し、最終的にニューラルネットワークモデルを作成する
input_tensor = Input(shape=(image_size,image_size,3))

# 転移学習のモデルとしてVGG16を使用
# 転移学習のために事前訓練済みのVGG16モデルをロードし、新しい入力テンソルを指定してモデルのトップ層（分類層）を省略する設定をしている。
# include_top=False: これは「上の部分（最終結果を出す部分）は使わない」という意味。自分たちで新しい上の部分を作るため。
vgg16 = VGG16(include_top = False, weights="imagenet",input_tensor=input_tensor)


# モデルの定義
# 自作モデルの作成
# 入力データをフラット化: Flatten 層で入力を1次元ベクトルに変換。
# 全結合層とシグモイド活性化関数: 256, 64, 32 個のニューロンを持つ全結合層にシグモイド活性化関数を適用し、中間層として特徴量を抽出。
# Dropout 層: 過学習を防ぐため、各全結合層の後に Dropout 層を追加。
# 出力層: 最後に 5 クラスの分類問題に対してソフトマックス関数を用いて確率を出力。
top_model = Sequential()
top_model.add(Flatten(input_shape=vgg16.output_shape[1:]))
top_model.add(Dense(256,activation="sigmoid"))
top_model.add(Dropout(0.5))
top_model.add(Dense(64,activation="sigmoid"))
top_model.add(Dropout(0.5))
top_model.add(Dense(32,activation="sigmoid"))
top_model.add(Dropout(0.5))
top_model.add(Dense(3,activation="softmax"))

#vggと top_modelの連結
model = Model(vgg16.inputs,top_model(vgg16.output))

# vgg16による特徴抽出部分の重みを15層までに固定（以降に新しい層(top_model)が追加
for layer in model.layers[:15]:
    layer.trainable = False

# コンパイルする(モデルがどうやって学習するかを決めること)
# 高水準プログラミング言語で書かれたソースコードを、コンピュータが直接実行できる機械語（バイナリコード）に変換するプロセスの
model.compile(loss="categorical_crossentropy",#モデルがどれだけ間違っているか
              optimizer=optimizers.SGD(learning_rate=1e-4,momentum=0.9),#モデルがどうやって間違いを直すか
              metrics=["accuracy"])


# 学習の実行
history = model.fit(X_train,y_train,batch_size=64,epochs=100,validation_data=(X_test,y_test))
model.save(r"C:\Users\admin\Desktop\VGG\model/weed2_eposhs100.h5")


# モデルの精度評価
scores = model.evaluate(X_test,y_test,verbose=1)
print("Test loss:",scores[0])
print("Test accuracy:",scores[1])

plt.plot(history.history["accuracy"],label="accuracy",ls="-",marker="o")
plt.plot(history.history["val_accuracy"],label="val_accuracy",ls="-",marker="x")
plt.ylabel("loss")
plt.xlabel("epoch")
plt.legend(loc="best")
plt.show()

sleep(3)

# 雑草ごとのF値
# F1スコアを計算するために予測を取得
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

# 各クラスごとのF1スコアを計算
f1_scores = f1_score(y_true_classes, y_pred_classes, average=None)

# クラスごとのF1スコアを表示
weed_list = ["birodomozuika",
             "himeodorikosou",
             "yagurumagiku"]

for i, score in enumerate(f1_scores):
    weed = weed_list[i]
    print(f"{weed}: {score}")
