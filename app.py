#import Flask
from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
from transformers import pipeline, DistilBertTokenizer
import joblib, os
import numpy as np
import tensorflow as tf
from tensorflow import keras

UPLOAD_FOLDER = 'static/uploads/'

#create an instance of Flask
app = Flask(__name__)
app.secret_key = "secret key"

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/', methods=['POST', 'GET'])
#predict
@app.route('/predict/', methods=['GET','POST'])
def predict():
    
    if request.method == "POST":
        
        #get form data
        question = request.form.get('question')
      
        #call preprocessDataAndPredict and pass inputs
        try:
            prediction = preprocessDataAndPredict(question)
            #pass prediction to template
            return render_template('predict.html', prediction = prediction, question = question)
   
        except ValueError:
            return "Please Enter A Valid Question"
  
        pass
    pass

@app.route('/predictimg/', methods=['GET','POST'])
def predictimg():
    
    if request.method == "POST":
        
        #get form data
        image = request.files['image']
        image.save(os.path.join('static/uploads', image.filename))

        #call preprocessDataAndPredict and pass inputs
        try:
            prediction = preprocessImageAndPredict(image)
            #pass prediction to template
            return render_template('predictimg.html', prediction = prediction, image = image, filename = image.filename)
   
        except ValueError:
            return "Please Upload A Valid Image"
  
        pass
    pass

def preprocessDataAndPredict(question):
    
    #keep all inputs in array
    test_data = {
        'question': question,
        'context': 'BTS has a lot of meanings – the original one being “Bangtan Sonyeondan” which means “Bulletproof Boy Scouts” in Korean. It signifies deflecting stereotypes, criticisms, and expectations like bullets. The acronym now also stands for “Beyond the Scene.” BTS consists of seven male members – Jin, Suga, J-Hppe, RM, V, Jungkook, and Jimin. The “leader” is RM. This popular K-Pop group wasn’t self-created – an entertainment company called Big Heat Entertainment held auditions to recruit members starting in 2010. The ultimate group wasn’t finalized until 2013. BTS started gaining attention six months before their debut due to their presence on social media and song covers. In March 2016, Forbes listed BTS as the most re-tweeted artist. BTS released 29 music videos and 22 singles in just 6 years from 2013 to 2018. The first album released by BTS was “Dark and Wild” which came out on August 19, 2014. BTS put on their own variety show which they starred in called “Rookie King: Channel Bangtan.” As of July 2018, BTS has won 53 awards since their formation. BTS is known for their lyrics that touch on sensitive topics like school bullying, societal ideals, mental health issues, suicide, nihilism, and female empowerment. RM’s initials stand for “Rap Monster,” but his full name is actually Kim Nam-joon. RM learned English by watching the popular American sitcom “Friends.” RM is also the only member who can carry a conversation in English. BTS member, Suga, is known for rapping in the band, but he also is proficient at the piano. Jungkook is the youngest member of the group, born on September 1st, 1997. Fans of BTS are called “ARMY” which is an acronym for “Adorable Representative M.C. for Youth.” Unlike most K-Pop bands, BTS hold a leading role in producing most of their own music. Many people believe this is an attribution to their success. BTS performed at the 2017 American Music Awards which was their first ever TV performance in the United States. BTS was the first K-Pop group to get their own Twitter emojis. The emoji was established as a challenge for BTS to find where their biggest fanbases were located. In the end, Brazil, Turkey, and Russia came out on top. BTS has sold at least 7 million albums globally as of July 2018. The third full-length album from BTS called “Love Yourself: Tear” launched at number one on the Billboard 200, which made them the only K-Pop group to do so. Their album “Love Yourself: Her” retailed more than 1.2 million copies on South Korea’s Gaon Album chart in its first month of release. It became the biggest selling Korean album by month in 16 years.'
    }
    print(question)
    
    #open file
    file = open("./model/qa_model.pkl","rb")
    
    #load trained model
    trained_model = joblib.load(file)
    tokenizer = DistilBertTokenizer.from_pretrained("./tokenizer/")
    
    #predict
    nlp = pipeline('question-answering', model=trained_model, tokenizer=tokenizer)
    prediction = nlp(test_data)
    print(prediction)
    return prediction['answer']
    pass

def preprocessImageAndPredict(image):
    
    #keep all inputs in array
    img = keras.preprocessing.image.load_img(
        os.path.join('static/uploads', image.filename), target_size=(180, 180)
    )
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch
    
    #open file
    # file = open("./model/btsclassifier.h5","rb")
    
    #load trained model
    trained_model = tf.keras.models.load_model("./model/btsclassifier.h5")
    
    #predict
    predictions = trained_model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    print(score)
    class_names = ['J-Hope', 'Jimin', 'Jin', 'Jungkook', 'RM', 'Suga', 'V']
    return class_names[np.argmax(score)]
    pass

if __name__ == '__main__':
    app.run(debug=True)