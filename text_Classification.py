from tkinter.font import Font
from sklearn.metrics import classification_report
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tkinter import *
from tkinter import font as tkFont  # for convenience
from tkinter import filedialog
from underthesea import word_tokenize
import regex as re
import os
############################## xây dựng hàm tiền xử lý văn bản ########################################


def text_preprocess(document):
    # tách từ
    document = word_tokenize(document, format="text")
    # đưa về lower
    document = document.lower()
    # xóa các ký tự không cần thiết
    document = re.sub(
        r'[^\s\wáàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệóòỏõọôốồổỗộơớờởỡợíìỉĩịúùủũụưứừửữựýỳỷỹỵđ_]', ' ', document)
    # xóa khoảng trắng thừa
    document = re.sub(r'\s+', ' ', document).strip()
    return document


############################## Thống kê các word xuất hiện ở tất cả các nhãn ########################################

f = open('news_categories.txt', mode='r', encoding='utf-8')
# data=f.read(1000000)
all_label = 18
count_lable_in_word = {}
count_word_in_label = {}
for line in f:
    words = line.split()  # tách từ
    label = words[0]  # lấy ra nhãn
#     print(words[:10000])
# #     print(label)
    if label not in count_word_in_label:
        count_word_in_label[label] = {}
# print(count_word_in_label)
    for word in words[1:]:
        count_word_in_label[label][word] = count_word_in_label[label].get(
            word, 0) + 1
# print((count_word_in_label['__label__thể_thao']))
        if word not in count_lable_in_word:
            count_lable_in_word[word] = set()
        count_lable_in_word[word].add(label)
# print(count_lable_in_word['chelsea'])
count = {}
for word in count_lable_in_word:
    if len(count_lable_in_word[word]) == all_label:
        count[word] = min([count_word_in_label[x][word]
                           for x in count_word_in_label])
#         print(count[word],word)
sorted_count = sorted(count, key=count.get, reverse=True)
# for word in sorted_count[:100]:
#     print(word, count[word])
############################## loại stopword khỏi dữ liệu ########################################


stopword = set()
with open('stopwords.txt', 'w', encoding='utf-8') as f:
    for word in sorted_count[:100]:
        stopword.add(word)
        f.write(word + '\n')
# # f=open ( 'stopwords.txt' , mode = 'r' , encoding = 'utf-8' )
# # print(f.read())


def remove_stopwords(line):
    words = []
    for word in line.strip().split():

        if word not in stopword:
            words.append(word)
    return ' '.join(words)


############################## Chia tập train/test ########################################

test_percent = 0.2
handle_news_categories = open(
    'handle_news_categories.txt', mode='r', encoding='utf-8')
text = []
label = []

for line in handle_news_categories:
    words = line.strip().split()
    label.append(words[0])
    text.append(' '.join(words[1:]))

X_train, X_test, y_train, y_test = train_test_split(
    text, label, test_size=test_percent, random_state=42)

label_encoder = LabelEncoder()
# print(list(label_encoder.classes_))
y_train = label_encoder.fit_transform(y_train)
y_test = label_encoder.fit_transform(y_test)

##############################  code tkinter ########################################


def openFile():
    tf = filedialog.askopenfilename(
        initialdir="E:\\DataMining\\code_thinter",
        title="Open Text file",
        filetypes=(("Text Files", "*.txt"),)
    )
    # pathh.insert(END, tf)
    tf = open(tf, 'r', encoding='utf-8')  # or tf = open(tf, 'r')
    data = tf.read()
    txtarea.insert(END, data)
    tf.close()


def predict():
    SVM_model = pickle.load(open(os.path.join("svm.pkl"), 'rb'))
    dataDemo = open('demo.txt', mode='r', encoding='utf-8')
    # print(f.read())
    with open('handle_demo.txt', 'w', encoding='utf-8') as hd:
        for line in dataDemo:
            line = text_preprocess(line)
            line = remove_stopwords(line)
            hd.write(line + '\n')
    handle_demo = open('handle_demo.txt', mode='r', encoding='utf-8')
    with open('handle_predict_demo.txt', 'w', encoding='utf-8') as hpd:
        for line in handle_demo:
            line = SVM_model.predict([line])
            line = label_encoder.inverse_transform(line)
            out_array = np.array_str(line)  # chuyển từ np.ndarray về str
            hpd.write(out_array + '\n')
    # or tf = open(tf, 'r')
    d = open('handle_predict_demo.txt', 'r', encoding='utf-8')
    data = d.read()
    txtPredict.insert(END, data)
    d.close()


root = Tk()
root.title('Python Guides')
scrW = root.winfo_screenwidth()
scrH = root.winfo_screenheight()
# root.geometry('1000x600+%d+%d' % (scrW/2-500, scrH/2-300))
root.geometry("%dx%d" % (scrW, scrH))
fontSize = tkFont.Font(family='Helvetica', size=20, weight=tkFont.BOLD)

frame2 = Frame(root)
# Create a Label in frame2
Button(
    frame2,
    text="Open File",
    command=openFile
).grid(ipadx=50, ipady=10, row=0, column=0, sticky=W, padx=20, pady=20)
Button(
    frame2,
    text="Predict",
    command=predict
).grid(ipadx=58, ipady=10, row=1, column=0, sticky=W, padx=20, pady=20)
# Create an Entry Widget in Frame2
myFont = Font(family="Times New Roman", size=20)
txtarea = Text(frame2, height=10, width=110)
txtarea.grid(row=0, column=1, sticky=W, padx=20, pady=20)
txtarea.configure(font=myFont)


txtPredict = Text(frame2, height=10, width=110)
txtPredict.grid(row=1, column=1, sticky=W, padx=20, pady=20)
txtPredict.configure(font=myFont)
frame2.pack()


root.mainloop()
