import os
from flask import Flask, request, redirect, render_template, flash
from werkzeug.utils import secure_filename
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.preprocessing import image

import numpy as np


classes = ["0","1","2","3","4","5","6","7","8","9"]
#分類したいクラス名をclassesリストに格納しておきましょう。今回は数字を分類するので0~9としておきます。
image_size = 28#image_sizeには学習に用いた画像のサイズを渡しておきます。今回はMNISTのデータセットを用いたので28とします。

UPLOAD_FOLDER = "uploads"#UPLOAD_FOLDERにはアップロードされた画像を保存するフォルダ名を渡しておきます。
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])
#ALLOWED_EXTENSIONSにはアップロードを許可する拡張子を指定します。

app = Flask(__name__)#Flaskクラスのインスタンスを作成します。

def allowed_file(filename):#アップロードされたファイルの拡張子のチェックをする関数
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
#'.' in filenameは、変数filenameの中に.という文字が存在するかどうかです。
#filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONSは、
# 変数filenameの.より後ろの文字列がALLOWED_EXTENSIONSのどれかに該当するかどうかです。
# rsplit()は基本的には split()と同じです。しかし、split()は区切る順序は文字列の最初から
# でしたが、rsplitは区切る順序が文字列の最後からになります。lower()は文字列を小文字に変換します。


model = load_model('./model.keras')#学習済みモデルをロード

#GETやPOSTはHTTPメソッドの一種です。GET はリソースを取り込むこと(ページにアクセスしたときにhtmlファイルを取り込む)、
# POST はデータをサーバーへ送信することを表します。
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('ファイルがありません')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('ファイルがありません')
            return redirect(request.url)
        #requestはウェブ上のフォームから送信したデータを扱うための関数であり、
        # request.methodにはリクエストのメソッドが格納されています。
        #ここでは、POSTリクエストにファイルデータが含まれているか、
        #また、ファイルにファイル名があるかをチェックします。
        #redirect()は引数に与えられたurlに移動する関数であり,
        #request.urlにはリクエストがなされたページのURLが格納されています。
        

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(UPLOAD_FOLDER, filename)) 
            filepath = os.path.join(UPLOAD_FOLDER, filename)
        #アップロードされたファイルの拡張子をチェックします。
        #ファイル名に危険な文字列がある場合に無効化（サニタイズ）します。
        #os.path.join()で引数に与えられたパスをosに応じて
        #結合(Windowsでは￥で結合し、Mac,Linaxでは/で結合する) しており、そのパスにアップロードされた画像を保存します。
        # また、その保存先をfilepathに格納します。
           
            img = image.load_img(filepath, color_mode='grayscale', target_size=(image_size,image_size))
            img = image.img_to_array(img)
            data = np.array([img])
            #受け取った画像を読み込み、np形式に変換
            #image.load_imgという画像のロードとリサイズを同時にできる関数を用います。
            # 引数には読み込みたい画像のURLと、その画像をリサイズしたいサイズを指定します。
            # さらに grayscale=Trueと渡すことで、モノクロで読み込むことができます。
            # image.img_to_arrayは引数に与えられた画像をNumpy配列に変換します

            #変換したデータをモデルに渡して予測する
            result = model.predict(data)[0]
            predicted = result.argmax()
            pred_answer = "これは " + classes[predicted] + " です"

            return render_template("index.html",answer=pred_answer)
            #render_templateの引数にanswer=pred_answerと渡すことで、
            # index.htmlに書いたanswerにpred_answerを代入することができます。

    return render_template("index.html",answer="")
    #POSTリクエストがなされないとき（単にURLにアクセスしたとき）にはindex.htmlのanswerには
    # 何も表示しません。

#自分のパソコンだけで動かす
#if __name__ == "__main__":
    app.run()
    #最後にapp.run()が実行され、サーバが立ち上がります。


#デプロイ
if __name__ == "__main__":
    port = int(os.environ.get('PORT', 8080))
    app.run(host ='0.0.0.0',port = port)