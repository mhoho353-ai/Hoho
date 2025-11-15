import streamlit as st
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

st.title("🎨 مولد لوحات الألوان الذكي")
st.write("ارفع صورة وسنستخرج منها أجمل 5 ألوان 💫")

uploaded = st.file_uploader("📂 ارفع صورتك هنا", type=["jpg", "png", "jpeg"])

if uploaded:
    image = Image.open(uploaded)
    st.image(image, caption="📷 الصورة الأصلية", use_container_width=True)

    img_np = np.array(image)
    img_np = img_np.reshape((-1, 3))

    kmeans = KMeans(n_clusters=5, n_init=10)
    kmeans.fit(img_np)
    colors = kmeans.cluster_centers_.astype(int)

    st.subheader("🎨 لوحة الألوان:")
    fig, ax = plt.subplots(1, 5, figsize=(10, 2))
    for i, color in enumerate(colors):
        ax[i].imshow([[color / 255]])
        ax[i].axis("off")
    st.pyplot(fig)

    st.subheader("🧾 رموز الألوان:")
    for color in colors:
        hex_code = '#%02x%02x%02x' % tuple(color)
        st.code(hex_code)
