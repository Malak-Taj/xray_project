import streamlit as st
from PIL import Image
import numpy as np
import time
import cv2
import tensorflow as tf
from tensorflow.keras import layers, models, Model, Input
from tensorflow.keras.layers import *
import os
# -------------------------------


# PAGE CONFIGURATION
st.set_page_config(page_title="X-ray Classifier", layout="wide")
st.markdown("<h1 style='text-align:center; color:#0A74DA;'>Chest X-ray Deep Learning App</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; font-size:25px;'>AI-powered Classification for lung health insights</p>", unsafe_allow_html=True)
st.write("""
    This deep learning model classifies **chest X-ray images**
    into categories such as *Normal*, *Pneumonia*, and *Tuberculosis*.

    **Model Highlights:**
    - CBAM (Convolutional Block Attention Module)
    - Focal Loss for imbalanced datasets
    - Grad-CAM for explainability
""")


# SIDEBAR
st.sidebar.header("Upload an X-ray Image")

uploaded_file = st.sidebar.file_uploader(
    "Choose an X-ray file (PNG/JPG/JPEG)",
    type=["png","jpg","jpeg"],
    help="Max size: 200MB"
)

with st.sidebar.expander("X-ray Image Tips"):
    st.markdown("""
    - Ensure X-ray is centered and clear.
    - Large images will be resized automatically.
    - Supported formats: PNG, JPG, JPEG
""")

progress_text = st.sidebar.empty()
progress_bar = st.sidebar.progress(0)


# CBAM ARCHITECTURE FUNCTION
def cbam_block(input_feature, ratio=8):
    channel = input_feature.shape[-1]

    # Channel Attention
    avg_pool = GlobalAveragePooling2D()(input_feature)
    max_pool = GlobalMaxPooling2D()(input_feature)
    avg_pool = Dense(channel // ratio, activation='relu')(avg_pool)
    max_pool = Dense(channel // ratio, activation='relu')(max_pool)
    avg_pool = Dense(channel, activation='sigmoid')(avg_pool)
    max_pool = Dense(channel, activation='sigmoid')(max_pool)
    channel_attention = Add()([avg_pool, max_pool])
    channel_attention = Activation('sigmoid')(channel_attention)
    channel_attention = Reshape((1, 1, channel))(channel_attention)
    channel_refined = Multiply()([input_feature, channel_attention])

    # Spatial Attention
    avg_pool = Lambda(lambda x: tf.reduce_mean(x, axis=-1, keepdims=True))(channel_refined)
    max_pool = Lambda(lambda x: tf.reduce_max(x, axis=-1, keepdims=True))(channel_refined)
    concat = Concatenate(axis=-1)([avg_pool, max_pool])
    spatial_attention = Conv2D(filters=1, kernel_size=7, padding='same', activation='sigmoid')(concat)
    spatial_refined = Multiply()([channel_refined, spatial_attention])

    return spatial_refined


# CBAM MODEL ARCHITECTURE
def build_cbam_model():
    img_input_ = Input(shape=(512, 512, 3))
    img_ = Resizing(224, 224)(img_input_)
    img_ = BatchNormalization()(img_)
    hidden = Conv2D(32, 3, padding='same', kernel_initializer='he_normal')(img_)
    hidden = BatchNormalization()(hidden)
    hidden = ReLU()(hidden)
    hidden = Conv2D(32, 3, padding='same', kernel_initializer='he_normal')(hidden)
    hidden = BatchNormalization()(hidden)
    hidden = ReLU()(hidden)
    hidden = MaxPool2D()(hidden)
    hidden = Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(hidden)
    hidden = BatchNormalization()(hidden)
    hidden = ReLU()(hidden)
    hidden = Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(hidden)
    hidden = BatchNormalization()(hidden)
    hidden = ReLU()(hidden)
    hidden = MaxPool2D()(hidden)
    hidden = Conv2D(128, 3, padding='same', kernel_initializer='he_normal', kernel_regularizer=tf.keras.regularizers.l2(1e-4))(hidden)
    hidden = BatchNormalization()(hidden)
    hidden = ReLU()(hidden)
    hidden = Conv2D(128, 3, padding='same', kernel_initializer='he_normal', kernel_regularizer=tf.keras.regularizers.l2(1e-4))(hidden)
    hidden = BatchNormalization()(hidden)
    hidden = ReLU()(hidden)
    hidden = cbam_block(hidden)
    hidden = GlobalAveragePooling2D()(hidden)
    hidden = Dense(128, kernel_initializer='he_normal')(hidden)
    hidden = BatchNormalization()(hidden)
    hidden = ReLU()(hidden)
    hidden = Dropout(0.4)(hidden)
    output = Dense(3, activation='softmax', kernel_initializer='glorot_normal')(hidden)
    return Model(inputs=[img_input_], outputs=[output])

# U-NET MODEL FUNCTION
def unet_small(input_size=(512,512,1)):
    inputs = layers.Input(input_size)
    c1 = layers.Conv2D(32,3,activation='relu',padding='same')(inputs)
    c1 = layers.Conv2D(32,3,activation='relu',padding='same')(c1)
    p1 = layers.MaxPooling2D((2,2))(c1)
    c2 = layers.Conv2D(64,3,activation='relu',padding='same')(p1)
    c2 = layers.Conv2D(64,3,activation='relu',padding='same')(c2)
    p2 = layers.MaxPooling2D((2,2))(c2)
    c3 = layers.Conv2D(128,3,activation='relu',padding='same')(p2)
    c3 = layers.Conv2D(128,3,activation='relu',padding='same')(c3)
    p3 = layers.MaxPooling2D((2,2))(c3)
    c4 = layers.Conv2D(256,3,activation='relu',padding='same')(p3)
    c4 = layers.Conv2D(256,3,activation='relu',padding='same')(c4)
    p4 = layers.MaxPooling2D((2,2))(c4)
    c5 = layers.Conv2D(512,3,activation='relu',padding='same')(p4)
    c5 = layers.Conv2D(512,3,activation='relu',padding='same')(c5)
    u6 = layers.Conv2DTranspose(256,(2,2),strides=(2,2),padding='same')(c5)
    u6 = layers.concatenate([u6, c4])
    c6 = layers.Conv2D(256,3,activation='relu',padding='same')(u6)
    c6 = layers.Conv2D(256,3,activation='relu',padding='same')(c6)
    u7 = layers.Conv2DTranspose(128,(2,2),strides=(2,2),padding='same')(c6)
    u7 = layers.concatenate([u7, c3])
    c7 = layers.Conv2D(128,3,activation='relu',padding='same')(u7)
    c7 = layers.Conv2D(128,3,activation='relu',padding='same')(c7)
    u8 = layers.Conv2DTranspose(64,(2,2),strides=(2,2),padding='same')(c7)
    u8 = layers.concatenate([u8, c2])
    c8 = layers.Conv2D(64,3,activation='relu',padding='same')(u8)
    c8 = layers.Conv2D(64,3,activation='relu',padding='same')(c8)
    u9 = layers.Conv2DTranspose(32,(2,2),strides=(2,2),padding='same')(c8)
    u9 = layers.concatenate([u9, c1])
    c9 = layers.Conv2D(32,3,activation='relu',padding='same')(u9)
    c9 = layers.Conv2D(32,3,activation='relu',padding='same')(c9)
    outputs = layers.Conv2D(1,(1,1),activation='sigmoid')(c9)
    return models.Model(inputs, outputs)


# SAFE LOAD FUNCTION
@st.cache_resource
def load_models():
    # --- U-Net ---
    seg_model = unet_small()
    seg_weights_path = "models/cxr_reg_weights.best.hdf5"
    if os.path.exists(seg_weights_path):
        seg_model.load_weights(seg_weights_path)
    else:
        st.error(f"U-Net weights not found at {seg_weights_path}")

    # --- CBAM ---
    cbam_model = build_cbam_model()
    cbam_weights_path = "models/model_cbam_last.weights.h5"  # <-- fixed extension
    if os.path.exists(cbam_weights_path):
        try:
            cbam_model.load_weights(cbam_weights_path)
        except Exception as e:
            st.error(f"Failed to load CBAM weights: {e}")
    else:
        st.error(f"CBAM weights not found at {cbam_weights_path}")

    return seg_model, cbam_model


# PREDICTION FUNCTION WITH SAFETY
def predict_image(img_array, seg_model, cbam_model):
    labels = ["Normal", "Pneumonia", "Tuberculosis"]
    try:
        gray_input = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        gray_input = cv2.resize(gray_input, (512,512)).astype(np.float32)/255.0
        gray_input = np.expand_dims(gray_input, axis=(0,-1))
        mask = seg_model.predict(gray_input)[0]
        mask = (mask > 0.5).astype(np.float32)
        masked = gray_input[0] * mask
        masked_input = np.expand_dims(masked, axis=0)
        masked_input = np.repeat(masked_input, 3, axis=-1)
        preds = cbam_model.predict(masked_input)[0]
        pred_idx = np.argmax(preds)
        return labels[pred_idx], preds[pred_idx], labels, preds
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        return None, None, labels, None


# LOAD MODELS
seg_model, cbam_model = load_models()


# MAIN APP
if uploaded_file is not None:
    for i in range(1, 101):
        progress_text.text(f"Processing: {i}%")
        progress_bar.progress(i)
        time.sleep(0.01)

    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)

    pred_label, confidence, labels, probs = predict_image(img_array, seg_model, cbam_model)
    if pred_label is not None:
        st.subheader("Prediction Result")
        bar_width = int(confidence * 100)
        st.markdown(f"""
        <div style='display:flex; align-items:center; margin-bottom:5px;'>
            <div style='flex:1; background-color:#e0e0e0; height:25px; border-radius:5px; margin-right:10px;'>
                <div style='width:{bar_width}%; background-color:#0A74DA; height:100%; border-radius:5px;'></div>
            </div>
            <div style='min-width:70px; font-weight:bold;'>{pred_label} ({confidence*100:.1f}%)</div>
        </div>
        """, unsafe_allow_html=True)



else:
    st.info("Upload an X-ray image from the sidebar to begin.")


# FOOTER
st.markdown("---")
st.markdown("""
<div style='text-align:center; color:gray; font-size:14px;'>
    © 2025 All Rights Reserved
</div>
""", unsafe_allow_html=True)
