import streamlit as st
from tensorflow.keras.applications.resnet50 import preprocess_input, ResNet50
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer
import numpy as np
import os
import time
import cv2
from PIL import Image

# Set the page configuration
st.set_page_config(
    page_icon=":smiley:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Function to capture image from webcam using OpenCV
def capture_image(label, save_folder):
    cam_port = 0  # Change this if you have multiple cameras
    cam = cv2.VideoCapture(cam_port)
    result, img = cam.read()
    cam.release()

    if result:
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        image_name = f"{label}_captured_{time.time()}.png"
        image_path = os.path.join(save_folder, image_name)
        cv2.imwrite(image_path, img)
        st.success(f"Image captured and saved in {save_folder}")
    else:
        st.error("Failed to capture image. Please try again.")

# me_files = st.file_uploader("Upload 'me' Class Images", type=["jpg", "png"], accept_multiple_files=True, key="me")
# not_me_files = st.file_uploader("Upload 'not me' Class Images", type=["jpg", "png"], accept_multiple_files=True, key="not_me")



st.sidebar.title("Navigation")
section = st.sidebar.radio("Go to", ["Introduction", "Section 2", "Section 3", "Section 4", "Section 5", "Section 6", "Section 7", "Section 8", "Section 9", "Section 10", "Section 11", "Section 12", "Section 13", "Section 14"])

# Set the theme to light
st.markdown(
    """
    <style>
        body {
        }

        p{
            text-align: justify;
            text-justify: inter-word;
        }
        .center {
            display: flex;
            justify-content: center;
            align-items: center;
            # height: 100vh;
            .video{
            width: 50%
            }
        h2{
        width:100%
        text-align: center;
        }
            
    </style>
    """,
    unsafe_allow_html=True,
)




# Section 1: What is Face Recognition? /////////////////////////////////////////////////////////////////////////////////////
if section == "Introduction":
    st.markdown('<div class="center"><h1>Face Recognition App</h1></div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<h2 class="header"> What is Face Recognition? </h2> ', unsafe_allow_html=True)
        st.markdown("""
                <div class=contaienr> <p>Facial Recognition is a way of recognizing a human face using biometrics.It consists of comparing features of a person’s face with a database of known faces to find a match. When the match is found correctly, the system is said to have ‘recognized’ the face. Face Recognition is used for a variety of purposes, like unlocking phone screens, identifying criminals,
            and authorizing visitors. </p> </div>
            """, unsafe_allow_html=True)

    with col2:
    # st.markdown('<img src="media/Picture1.png">', unsafe_allow_html=True)
        image1 = Image.open('media/Picture1.png')
        st.image(image1, caption='')
    if st.button("Next"):
        st.markdown("<a href='#Section-2'>Go to Section 2</a>", unsafe_allow_html=True)

  




# Section 2: How do Computers Recognize Faces? /////////////////////////////////////////////////////////////////////////////////////
elif section == "Section 2":
    st.markdown('<div class="center"><h2>Section 2: How do Computers Recognize Faces?</h2></div>', unsafe_allow_html=True)
    st.markdown("""
                <div class=contaienr> <p>The Face Recognition system uses Machine Learning to analyze and process facial features from images or videos. Features can include anything, from the distance between your eyes to the size of your nose. These features, which are unique to each person, are also known as Facial Landmarks. The machine learns patterns in these landmarks by training Artificial Neural Networks. The machine can then identify people’s faces by matching these learned patterns against new facial data.
</p> </div>
""", unsafe_allow_html=True)
    
    # video_path = "media/next_for_fr.mp4"

    # css_code = """
    #     <style>
    #         .video {
    #             display: flex;
    #             justify-content: center;
    #             align-items: center;
    #             height: 100vh;
    #         }
    #         .video-container {
    #             width: 50%;
    #         }
    #     </style>
    # """
    
    # # st.markdown(css_code, unsafe_allow_html=True)
    # st.markdown(f'<div class="video"><div class="video-container">{st.video(video_path)}</div></div>', unsafe_allow_html=True)


    video_url = "media/next_for_fr.mp4"
    st.video(video_url)
    
    if st.button("Previous"):
        st.markdown("<a href='#Section-1'>Go to Section 1</a>", unsafe_allow_html=True)
    if st.button("Next"):
        st.markdown("<a href='#Section-3'>Go to Section 3</a>", unsafe_allow_html=True)

elif section == "Section 3":
    st.markdown("<div class='center'><h2>Teach the Computer to Recognize your Face</h2></div>", unsafe_allow_html=True)
    
    columns = st.columns([1, 1, 1, 1, 1, 1, 1])

    # Step 1
    step1 = columns[0].button("Step 1", key="step1", help="Collect Data", on_click=None, args=None, kwargs=None)
    columns[1].write("\u2192", unsafe_allow_html=True)  # Unicode right arrow

    # Step 2
    step2 = columns[2].button("Step 2", key="step2", help="Train", on_click=None, args=None, kwargs=None)
    columns[3].write("\u2192", unsafe_allow_html=True)  # Unicode right arrow

    # Step 3
    step3 = columns[4].button("Step 3", key="step3", help="Test", on_click=None, args=None, kwargs=None)
    columns[5].write("\u2192", unsafe_allow_html=True)  # Unicode right arrow

    # Step 4
    step4 = columns[6].button("Step 4", key="step4", help="Export", on_click=None, args=None, kwargs=None)

    # Style for buttons
    button_style = """
        <style>
            .stButton button {
                background-color: #007BFF;
                color: white;
                font-weight: bold;
            }
        </style>
    """

    # Display the custom CSS for button styling
    st.markdown(button_style, unsafe_allow_html=True)

    # Labels for each step
    columns2 = st.columns([1, 1, 1, 1, 1, 1, 1])

    # Collect Data
    columns2[0].markdown("""<p style="text-align:center; font-weight: bold;">  Collect Data  </p>""", unsafe_allow_html=True)

    # Train
    columns2[2].markdown("""<p style="text-align:center; font-weight: bold;">     Train  </p> """, unsafe_allow_html=True)

    # Test
    columns2[4].markdown("""<p style="text-align:center; font-weight: bold;">       Test  </p>    """, unsafe_allow_html=True)

    # Export
    columns2[6].markdown("""<p style="text-align:center; font-weight: bold;">      Export  </p> """, unsafe_allow_html=True)

# section3: Collect data/////////////////////////////////////////////////////////////////////////////////////
elif section == "Section 4":
    st.markdown("<div class = 'center'><h2 id='Section-3'>Step 1 - Collect Data</h2></div>", unsafe_allow_html=True)
    st.markdown("""
                <div class=contaienr> We want our model to learn how to recognize your face. We will need two kinds of images for this - images of you, and images of people who are not you. This way, the model will learn to recognize how you look and also recognize how you don’t look. </div> """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
                <div class=contaienr> <p>Let’s start by giving the machine lots of images of you in different places, in different poses, and at different angles. </p> </div>""", unsafe_allow_html=True)

    with col2:
        st.markdown("""
                    <div class=contaienr> <p>Next, let’s give it images of people that are not you, so the machine understands the difference.</p>
                </div>""", unsafe_allow_html=True)


# Section 3: Teach the Computer to Recognize your Face /////////////////////////////////////////////////////////////////////////////////////
elif section == "Section 5":
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
                <div class=contaienr> <p>Let’s start by giving the machine lots of images of you in different places, in different poses, and at different angles. </p> </div>""", unsafe_allow_html=True)
        st.markdown(""" <h4>Upload 'me' images</h4>  """, unsafe_allow_html=True)
        me_files = st.file_uploader("", type=["jpg", "png"], accept_multiple_files=True, key="me")
        st.markdown(""" <h4>Capture 'me' images</h4>  """, unsafe_allow_html=True)
        if st.button("Capture 'me' Image"):
            capture_image('me', os.path.abspath('captured_images/me'))

    with col2:
        st.markdown("""
                    <div class=contaienr> <p>Next, let’s give it images of people that are not you, so the machine understands the difference.</p>
                </div>""", unsafe_allow_html=True)
        st.markdown(""" <h4>Upload 'not me' images</h4>  """, unsafe_allow_html=True)
        not_me_files = st.file_uploader("Upload 'not me' Class Images", type=["jpg", "png"], accept_multiple_files=True, key="not_me")
        st.markdown(""" <h4>Capture 'not me' images</h4>  """, unsafe_allow_html=True)
        if st.button("Capture 'not me' Image"):
            capture_image('not_me', os.path.abspath('captured_images/not_me'))

    
    if st.button("Previous"):
        st.markdown("<a href='#Section-3'>Go to Section 3</a>", unsafe_allow_html=True)

# /////////////////////////////////////////////////////////////////////////////////////
elif section == "Section 6":
    st.markdown('<div class="center"><h2 class="header"> Step 2 - Train The Machine </h2> </div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
                <div class=contaienr> <p>Next, we need to train the machine (or model) to recognize pictures of you. The model uses the samples of images you provided for this. This method is called “Supervised learning” because of the way you ‘supervised’ the training. The model learns from the patterns in the photos you’ve taken. It mostly takes into consideration the facial features or Facial Landmarks and associates the landmark of each face with the corresponding label.
                </p> </div>  """, unsafe_allow_html=True)

    with col2:
        # st.markdown('<img src="media/Picture1.png">', unsafe_allow_html=True)
        image1 = Image.open('media/Picture2.png')
        st.image(image1, caption='')
# sectino 7 /////////////////////////////////////////////////////////////////////////////////////
elif section == "Section 7":
    st.markdown('<div class = "center"><h2 class="header"> What do you mean by machine learning ?</h2></div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        
        st.markdown("""
                <div class=contaienr> <p>Machine learning is the process of making systems that learn and improve by themselves. The model learns from the data and makes predictions. It then checks with your label to see if it predicted the label correctly. If it didn’t, then it tries again. It keeps repeating this process with an aim to get better at the predictions.
                </p> </div>  """, unsafe_allow_html=True)

    with col2:
    # st.markdown('<img src="media/Picture1.png">', unsafe_allow_html=True)
        image1 = Image.open('media/Picture4.png')
        st.image(image1, caption='')

# section 8 /////////////////////////////////////////////////////////////////////////////////////
elif section == "Section 8":
    st.markdown('<div class = "center"><h2 class="header">How to Setup the Model for Training ?</h2> </div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
                <div class=contaienr> <p>Machine learning models can be of different types. One commonly used model is an Artificial Neural Network (ANN). 
                
                Neural networks mimic how the human brain works and interprets information. They consist of a many interconnected elements called Nodes. These nodes are organized in multiple layers, where nodes of each layer connect to nodes of the next layer.  

                </p> </div>  """, unsafe_allow_html=True)

    with col2:
    # st.markdown('<img src="media/Picture1.png">', unsafe_allow_html=True)
        image1 = Image.open('media/Picture5.png')
        st.image(image1, caption='')

    st.markdown('<div class="center"><h2 class="header">ANNs are Similar to our Brain </h2> </div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
                <div class=contaienr> <p>The basic unit of our brain’s system is a neuron. We have approximately 86 billion neurons in our brain. Each is connected to another neuron through connections called synapses.

                A neuron can pass an electrical impulse to another neuron that is connected to it. This neuron can further pass on another electrical impulse to yet another connected neuron. In this way, a complex network of neurons is created in the human brain.

                This same concept (of a network of neurons) is used in machine learning or ANNs. In this case, the neurons are artificially created in a machine learning system. When many such artificial neurons (nodes) are connected, it becomes an Artificial Neural Network.


                </p> </div>  """, unsafe_allow_html=True)

    with col2:
    # st.markdown('<img src="media/Picture1.png">', unsafe_allow_html=True)
        image1 = Image.open('media/Picture6.gif')
        st.image(image1, caption='')

    st.markdown("""
                <div class=contaienr> <p>Neurons in the first layer process signals that are input into the Neural network. They then send the results to connected neurons in the second layer. These results are then processed by neurons of the second layer and the results of this processing are sent to neurons of the third layer. This process continues till the signal reaches the last layer.

                The first layer of the Neural Network is called the input layer and, while the last layer is called the output layer. All layers in the middle comprise the hidden layers.

                </p> </div>  """, unsafe_allow_html=True)
    
    image1 = Image.open('media/Picture7.gif')
    st.image(image1, caption='')

elif section == "Section 9":
    st.markdown("""
                <div class=contaienr> <p>How your model trains depends on the Training Parameters that you set. Training parameters are values that control certain properties of the training process and of the resulting ML model. Let’s look at 2 important training parameters – epochs and learning rate, number of layers.
                </p> </div>  """, unsafe_allow_html=True)
    st.markdown("""  <h4>Epochs:</h4>  """, unsafe_allow_html=True)
    st.markdown("""
                <div class=contaienr> <p>In machine learning, "epochs" are the number of times the algorithm goes through the entire training dataset. It's like repeating a book or a song multiple times to remember it better. More epochs give the machine learning model more opportunities to learn from the data, and it can become more accurate. 
                However, too many epochs can also make the model memorize the data instead of learning from it, which isn't good.
                </p> </div>  """, unsafe_allow_html=True)
    st.markdown("""  <h4>Learning rate:</h4>  """, unsafe_allow_html=True)
    st.markdown("""
                <div class=contaienr> <p>Learning rate is how fast you want the model to learn during training. Think of it as how big a step the model takes when trying to improve itself. A high learning rate means big steps, and the model may overshoot the best solution. A low learning rate means small steps, and the model may take a long time to improve or may get stuck in a suboptimal solution. Finding the right learning rate is important because it affects how quickly and effectively the model learns.

                </p> </div>  """, unsafe_allow_html=True)
    st.markdown("""  <h4>Number of Hidden Layers: </h4>  """, unsafe_allow_html=True)
    st.markdown("""
                <div class=contaienr> <p>Every Neural Network has one input and one output layer, but can have any number of hidden layers. Machine Learning Engineers often use systematic experimentation to discover what works best for the specific data. They train the model with a different number of hidden layers to see which one works best.
                </p> </div>  """, unsafe_allow_html=True)

elif section == "Section 10":
    st.markdown('<h2 class="header"> Train the Machine </h2> ', unsafe_allow_html=True)
    st.write("""Now let us set up our Machine Learning model!  Enter the number of epochs for which you would like the model to train:""")
    epochs_duplicate = st.slider("Number of Epochs", 10, 100, 10)
    epochs = (epochs_duplicate // 10)
    st.write("""Once your model is all set, you can start training your model - 
    """)
    # Train the model
    if st.button("Train Model"):
        # Gather paths for uploaded and captured images
        me_folder = os.path.abspath('captured_images/me')
        not_me_folder = os.path.abspath('captured_images/not_me')

        # Process uploaded images
        processed_images = []
        labels = []

        for uploaded_file in me_files or []:
            img = image.load_img(uploaded_file, target_size=(224, 224))
            img = image.img_to_array(img)
            img = np.expand_dims(img, axis=0)
            img = preprocess_input(img)
            processed_images.append(img)
            labels.append(1)  # 'me' class

        for uploaded_file in not_me_files or []:
            img = image.load_img(uploaded_file, target_size=(224, 224))
            img = image.img_to_array(img)
            img = np.expand_dims(img, axis=0)
            img = preprocess_input(img)
            processed_images.append(img)
            labels.append(0)  # 'not me' class

        # Process captured images if the folders exist
        if os.path.exists(me_folder) and os.path.exists(not_me_folder):
            for img_filename in os.listdir(me_folder):
                img_path = os.path.join(me_folder, img_filename)
                img = image.load_img(img_path, target_size=(224, 224))
                img = image.img_to_array(img)
                img = np.expand_dims(img, axis=0)
                img = preprocess_input(img)
                processed_images.append(img)
                labels.append(1)

            for img_filename in os.listdir(not_me_folder):
                img_path = os.path.join(not_me_folder, img_filename)
                img = image.load_img(img_path, target_size=(224, 224))
                img = image.img_to_array(img)
                img = np.expand_dims(img, axis=0)
                img = preprocess_input(img)
                processed_images.append(img)
                labels.append(0)

        if processed_images:  # Check if any images are available for training
            X_train = np.vstack(processed_images)
            y_train = np.array(labels)

            # Rest of your training code...
        else:
            st.warning("No images available for training. Please capture or upload images.")

    
        # Train the model, save the model, etc.
        st.write(f"Training with {epochs_duplicate} epochs...")
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation='relu')(x)
        predictions = Dense(2, activation='softmax')(x)  # 2 classes: 'me' and 'not me'
        model = Model(inputs=base_model.input, outputs=predictions)

        for layer in base_model.layers:
            layer.trainable = False

        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        model.fit(X_train, y_train, epochs=epochs)  # Training with user-defined epochs

        # Calculate training accuracy
        train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
        st.write(f"Training complete! Training Accuracy: {train_acc * 100:.2f}%")

        # Save the model
        model.save('model.h5')

elif section == "Section 11":
        st.markdown('<h2 class="header"> Train the Machine Again </h2> ', unsafe_allow_html=True)
        st.write("""If the accuracy is not good enough you can consider re-adjusting the training parameters, and training again. """)
        epochs_duplicate = st.slider("Number of Epochs", 10, 100, 10)
        epochs = (epochs_duplicate // 10)
        st.write("""Once your model is all set, you can start training your model - 
        """)
        # Train the model
        if st.button("Start Training"):
            # Gather paths for uploaded and captured images
            me_folder = os.path.abspath('captured_images/me')
            not_me_folder = os.path.abspath('captured_images/not_me')

            # Process uploaded images
            processed_images = []
            labels = []

            for uploaded_file in me_files or []:
                img = image.load_img(uploaded_file, target_size=(224, 224))
                img = image.img_to_array(img)
                img = np.expand_dims(img, axis=0)
                img = preprocess_input(img)
                processed_images.append(img)
                labels.append(1)  # 'me' class

            for uploaded_file in not_me_files or []:
                img = image.load_img(uploaded_file, target_size=(224, 224))
                img = image.img_to_array(img)
                img = np.expand_dims(img, axis=0)
                img = preprocess_input(img)
                processed_images.append(img)
                labels.append(0)  # 'not me' class

            # Process captured images if the folders exist
            if os.path.exists(me_folder) and os.path.exists(not_me_folder):
                for img_filename in os.listdir(me_folder):
                    img_path = os.path.join(me_folder, img_filename)
                    img = image.load_img(img_path, target_size=(224, 224))
                    img = image.img_to_array(img)
                    img = np.expand_dims(img, axis=0)
                    img = preprocess_input(img)
                    processed_images.append(img)
                    labels.append(1)

                for img_filename in os.listdir(not_me_folder):
                    img_path = os.path.join(not_me_folder, img_filename)
                    img = image.load_img(img_path, target_size=(224, 224))
                    img = image.img_to_array(img)
                    img = np.expand_dims(img, axis=0)
                    img = preprocess_input(img)
                    processed_images.append(img)
                    labels.append(0)

            if processed_images:  # Check if any images are available for training
                X_train = np.vstack(processed_images)
                y_train = np.array(labels)

                # Rest of your training code...
            else:
                st.warning("No images available for training. Please capture or upload images.")

        
            # Train the model, save the model, etc.
            st.write(f"Training with {epochs_duplicate} epochs...")
            base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
            x = base_model.output
            x = GlobalAveragePooling2D()(x)
            x = Dense(1024, activation='relu')(x)
            predictions = Dense(2, activation='softmax')(x)  # 2 classes: 'me' and 'not me'
            model = Model(inputs=base_model.input, outputs=predictions)

            for layer in base_model.layers:
                layer.trainable = False

            model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
            model.fit(X_train, y_train, epochs=epochs)  # Training with user-defined epochs

            # Calculate training accuracy
            train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
            st.write(f"Training complete! Training Accuracy: {train_acc * 100:.2f}%")

            # Save the model
            model.save('model.h5')

elif section == "Section 12":
    st.markdown('<h2 class="header"> Test the model </h2> ', unsafe_allow_html=True)

    # Option to upload a test image
    st.markdown("<h3 class='sub-header'>Upload Test Image</h3>", unsafe_allow_html=True)
    test_image = st.file_uploader("Upload a test image...", type=["jpg", "png"])

    # Option to capture a test image
    st.markdown("<h3 class='sub-header'>Capture Test Image</h3>", unsafe_allow_html=True)
    if st.button("Capture Test Image"):
        capture_image('test_capture', 'captured_images/test_capture')

    # Process the uploaded or captured test image
    if test_image or os.path.exists('captured_images/test_capture'):
        st.write("Processing test image...")
        
        if os.path.exists('captured_images/test_capture'):
            # Use the captured test image
            test_image_path = os.path.join('captured_images/test_capture', os.listdir('captured_images/test_capture')[0])
        else:
            # Use the uploaded test image
            test_image_path = 'uploaded_test_image.png'
            with open(test_image_path, "wb") as f:
                f.write(test_image.read())

        img = image.load_img(test_image_path, target_size=(224, 224))
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)

        # Make prediction
        model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        x = model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation='relu')(x)
        predictions = Dense(2, activation='softmax')(x)  # 2 classes: 'me' and 'not me'
        model = Model(inputs=model.input, outputs=predictions)

        model.load_weights('model.h5')
        prediction = model.predict(img)
        predicted_class = np.argmax(prediction)

        # Display result
        if predicted_class == 1:
            st.write("Result: This is you!")
        else:
            st.write("Result: This is not you.")
    else:
        st.warning("Please upload a test image or capture one.")

elif section == "Section 13":
    st.markdown('<h4 class="header"> Was the model able to recognize the face correctly?</h4> ', unsafe_allow_html=True)


    selection = st.radio(
    "",
    ["YES", "NO"]
    )

    if selection == 'NO':
        st.markdown('<h2 class="header"></h2> ', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<h4 class="header">If not, then you have 2 options: </h4> ', unsafe_allow_html=True)
            st.markdown("""
                <div>
                        <ul>
                            <li>Adjust training parameters and Re-train the model</li>
                            <li>Add more images to your dataset</li>
                        </ul>
                        </div>


                """, unsafe_allow_html=True)

        with col2:
            st.markdown('<h4 class="header">How do you improve the accuracy of the system? </h4> ', unsafe_allow_html=True)
            st.markdown("""
                <div>
                        <ol>
                            <li>You can collect more data to train the system. The more data you feed into the system, the more exposed it is to a variety of
                            examples and the better the predictions. Try training the model with more images of each label.
                            </li>
                            <li>You can the fine-tune parameters of the machine learning model. Play around with the number of epochs, hidden layers, and the learning rate. Try different values and see which combination of parameters makes the model’s predictions more accurate. </li>
                        </ol>
                </div>


            """,  unsafe_allow_html=True)

    else:
        st.markdown('<h2 class="header">Great ! Go to next section.</h2> ', unsafe_allow_html=True)
