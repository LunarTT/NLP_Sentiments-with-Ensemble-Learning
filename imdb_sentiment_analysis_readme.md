# 🎬 IMDb Movie Reviews Sentiment Analysis & Machine Learning Pipeline

โปรเจกต์วิเคราะห์อารมณ์และจำแนกความรู้สึกจากข้อความรีวิวหนัง (Sentiment Analysis) บนชุดข้อมูล IMDb Dataset โดยประยุกต์ใช้การทำความสะอาดข้อความขั้นสูง (Advanced Text Cleansing & Lemmatization), การแปลงข้อเป็นเวกเตอร์หลายรูปแบบ (BoW, TF-IDF, Word2Vec, FastText, GloVe), การเปรียบเทียบประสิทธิภาพโมเดล Machine Learning 5 อัลกอริทึม ทั้งในรูปแบบ Train-Test Split (80/20) และ 5-Fold Cross Validation รวมถึงการนำโมเดลมารวมพลังด้วยเทคนิค Ensemble Learning (Hard Voting, Soft Voting, Weighted Ensemble)

## 📌 คุณสมบัติเด่นของโปรเจกต์ (Key Features)

1. **Advanced Text Preprocessing**:

   * ลบแท็ก HTML และขยายคำย่อ (Contractions Expansion) เช่น `can't` -> `cannot`

   * แปลงคำบอกคะแนนและอีโมจิเป็นสัญลักษณ์อารมณ์ เช่น `10/10` -> `perfect`, `:-)` -> `smile`

   * คัดเลือก Custom Stopwords โดย **คงไว้ซึ่งคำปฏิเสธ (Negation)** และ **คำเน้นระดับอารมณ์ (Intensifiers)** เช่น `not`, `never`, `very`, `extremely`, `but` เพื่อป้องกันการสูญเสียบริบททางอารมณ์

   * ทำ **POS-Guided Lemmatization** คืนรูปคำมูลฐานตามชนิดของคำด้วย NLTK WordNet

2. **Feature Representations (5 รูปแบบ)**:

   * **Bag of Words (BoW)** (Unigram + Bigram)

   * **TF-IDF Vectorizer** (Sublinear Scaling, Unigram + Bigram)

   * **Word2Vec** (`word2vec-google-news-300` - TF-IDF Weighted)

   * **FastText** (`fasttext-wiki-news-subwords-300` - TF-IDF Weighted)

   * **GloVe** (`glove-wiki-gigaword-300` - TF-IDF Weighted)

3. **Machine Learning Classifiers (5 โมเดลหลัก)**:

   * **ANN / MLPClassifier** (Artificial Neural Network)

   * **Random Forest Classifier**

   * **Linear SVM** (Support Vector Machine)

   * **Logistic Regression**

   * **Naive Bayes** (MultinomialNB สำหรับ BoW/TF-IDF และ GaussianNB สำหรับ Word Embedding)

4. **Comprehensive Evaluation**:

   * การวัดผลแบบ **80/20 Train-Test Split**

   * การวัดผลแบบ **5-Fold Stratified Cross-Validation**

   * รายงานผลด้วยค่า **Accuracy, Precision, Recall, F1-Score**

   * กราฟเปรียบเทียบผลลัพธ์: Confusion Matrix Heatmaps, Radar Charts, Bar Plots และ Line Charts

5. **Ensemble Learning Techniques**:

   * **Hard Voting Ensemble** (โหวตเสียงข้างมาก)

   * **Soft Voting Ensemble** (เฉลี่ยค่าความน่าจะเป็น)

   * **Weighted Ensemble** (ถ่วงน้ำหนักตามความเก่งของโมเดลด้วยค่า F1-Score)

## 📊 รายละเอียดชุดข้อมูล (Dataset)

* **Source**: IMDb Labelled Dataset (`imdb_labelled.txt`)

* **Total Samples**: 1,000 ข้อความรีวิวภาษาอังกฤษ

* **Data Balance**: สมดุล 50:50

  * `1` = **Positive** (ความรู้สึกเชิงบวก / ชื่นชอบ) - 500 ตัวอย่าง

  * `0` = **Negative** (ความรู้สึกเชิงลบ / ไม่ประทับใจ) - 500 ตัวอย่าง

* **Format**: File TSV (`\t` separated) ประกอบด้วย Column ข้อความรีวิว และ Label

## 🛠️ โครงสร้างไฟล์ใน Repositories

```
├── Dataset/
│   └── imdb_labelled.txt       # ชุดข้อมูล IMDb Labelled Dataset
├── sentiment_analysis.ipynb    # Jupyter / Colab Notebook หลัก
├── requirements.txt            # ไฟล์ระบุไลบรารีที่จำเป็น
└── README.md                   # อธิบายโปรเจกต์และวิธีการใช้งาน

```

## 🚀 วิธีการนำไปเล่น / การติดตั้งและเรียกใช้งาน (How to Run)

คุณสามารถเลือกเล่นและรันโปรเจกต์นี้ได้ 2 วิธีหลักๆ:

### วิธีที่ 1: รันบน Google Colab (แนะนำ 🌟)

1. **อัปโหลดไฟล์ไปยัง Google Drive**:

   * สร้างโฟลเดอร์ใน Google Drive ชื่อ `Colab Notebooks/Dataset`

   * นำไฟล์ `imdb_labelled.txt` ไปวางไว้ในโฟลเดอร์ `Dataset/`

2. **เปิดไฟล์ Notebook บน Google Colab**:

   * อัปโหลดไฟล์ `.ipynb` ขึ้น Google Colab

3. **เรียกใช้การเชื่อมต่อ Google Drive**:

   * รันเซลล์แรกๆ ใน Notebook เพื่อทำการ `drive.mount('/content/drive')`

4. **รันเซลล์ตามลำดับ**:

   * สามารถกด `Runtime` -> `Run all` เพื่อรันการทำงานทั้งหมดตั้งแต่ Data Preprocessing จนถึง Ensemble Learning

### วิธีที่ 2: รันบนเครื่องส่วนบุคคล (Local Machine)

#### 1. คลองโปรเจกต์ (Clone Repository)

```
git clone https://github.com/your-username/imdb-sentiment-analysis.git
cd imdb-sentiment-analysis

```

#### 2. สร้าง Virtual Environment (แนะนำ)

```
# บน Windows
python -m venv venv
venv\Scripts\activate

# บน macOS / Linux
python3 -m venv venv
source venv/bin/activate

```

#### 3. ติดตั้ง Dependencies

สร้างไฟล์ `requirements.txt` หากยังไม่มี หรือรันคำสั่งติดตั้งแพ็กเกจเหล่านี้:

```
pip install pandas numpy scikit-learn nltk gensim matplotlib seaborn jupyter

```

#### 4. เปิดใช้งาน Jupyter Notebook

```
jupyter notebook

```

เปิดไฟล์ `sentiment_analysis.ipynb` และปรับแก้ path ของชุดข้อมูลในโค้ดให้อ่านจากโฟลเดอร์ Local เช่น:

```
# เปลี่ยนจาก path ของ Colab เป็น Local Path
dataset = pd.read_csv('Dataset/imdb_labelled.txt', sep='\t', header=None, quoting=csv.QUOTE_NONE)

```

## 🔄 ขั้นตอนการประมวลผลโมเดล (Pipeline Overview)

```
flowchart TD
    A[Raw IMDb Dataset] --> B[Advanced Text Cleansing & Lemmatization]
    B --> C[Stratified Train-Test Split 80/20]
    C --> D[Feature Engineering]
    D --> D1[BoW & TF-IDF]
    D --> D2[Word2Vec, FastText, GloVe]
    D1 --> E[Model Training & CV]
    D2 --> E
    E --> F[Evaluation & Visualization]
    F --> G[Ensemble Learning - Voting & Weighted]

```

1. **Text Cleansing**: ลบตัวอักษรขยะ, ขยายคำย่อ, ปรับแต่ง Stopwords ให้เหลือคำปฏิเสธและคำแสดงอารมณ์, ทำ Lemmatization ด้วย POS Tagging

2. **Vectorization & Word Embeddings**: แปลงข้อความความยาวแปรผันเป็นเวกเตอร์ตัวเลขที่มีขนาดคงที่

3. **Model Training**: เทรนโมเดล 5 ประเภทผ่าน Feature Representations แต่ละแบบ

4. **Evaluation**:

   * สรุปตารางเปรียบเทียบ Accuracy, Precision, Recall, F1-Score

   * แสดง Confusion Matrix เพื่อดูการทายผิดพลาด (False Positive / False Negative)

   * วาด Radar Chart และ Trend Line Comparison

5. **Ensemble Learning**: คัดเลือกโมเดลที่มีประสิทธิภาพสูงสุด (Top Models) มารวมผลทำนายเพื่อเพิ่มความแม่นยำและความเสถียร

## 📈 สรุปผลการทดลอง (Key Results)

* **Feature Performance**: การใช้ **TF-IDF** ร่วมกับ **BoW** สำหรับโมเดลตระกูล Linear (เช่น Linear SVM และ Logistic Regression) ให้ประสิทธิภาพสูงในการแยกแยะข้อความสั้น

* **Word Embeddings**: การถ่วงน้ำหนักด้วย TF-IDF บน Pre-trained Word Embeddings (Word2Vec, FastText, GloVe) ช่วยจับความหมายเชิงบริบท (Semantic Context) ได้ดีเยี่ยม

* **Ensemble Advantage**: เทคนิค **Soft Voting** และ **Weighted Ensemble** ช่วยลดความเอนเอียง (Variance) ของโมเดลเดี่ยว และยกระดับค่า F1-Score รวมให้มีความเสถียรสูงขึ้น

## 📝 ไลเซนส์ (License)

โปรเจกต์นี้เปิดให้ใช้งานและศึกษาฟรีภายใต้สิทธิ์ [MIT License](LICENSE)