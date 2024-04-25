from sklearn import svm

data = [[170,58],[148,47],[168,70],[170,53],[154,49],[160,53],[171,75]]
data_sei = [1,2,1,2,2,2,1]
# 1男性,2女性

clf = svm.SVC(gamma=0.01)
#SVCが学習機、gammaがオプション
clf.fit(data,data_sei)
# 学習機にdataとdata_seiを渡す

data2 = [[174,59],[148,51],[168,57],[170,55]]
print(clf.predict(data2))
# 学習機にテスト用のデータを渡す
# 結果[1 2 2 2]
data2_sei = [1,2,2,1]
print(clf.score(data2,data2_sei))
# 正答率75%