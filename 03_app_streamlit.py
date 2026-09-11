import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Dự đoán chất lượng mạng 4G",
    page_icon="📶",
    layout="centered",
)

# ------------------------------------------------------------
# ĐIỀU HƯỚNG (SIDEBAR)
# ------------------------------------------------------------
st.sidebar.title("📶 Menu")
page = st.sidebar.radio(
    "Chọn trang",
    ["🔍 Dự đoán chất lượng mạng", "ℹ️ Giới thiệu & Hướng dẫn đo chỉ số"],
    key="nav_page",  # thêm key ổn định cho widget điều hướng
)

# ------------------------------------------------------------
# LOAD MÔ HÌNH
# ------------------------------------------------------------
@st.cache_resource
def load_model():
    model = joblib.load("best_random_forest_model.joblib")
    feature_cols = joblib.load("feature_columns.joblib")
    return model, feature_cols

QUALITY_LABELS = {
    "RATTOT": ("Rất tốt", "🟢"),
    "TOT": ("Tốt", "🟢"),
    "TB": ("Trung bình", "🟡"),
    "KEM": ("Kém", "🔴"),
}

# ==============================================================
# TRANG 1: DỰ ĐOÁN
# ==============================================================
if page == "🔍 Dự đoán chất lượng mạng":

    try:
        model, feature_cols = load_model()
    except FileNotFoundError:
        st.error(
            "Chưa tìm thấy mô hình đã huấn luyện. "
            "Hãy chạy '02_train_compare_models.py' trước để tạo file "
            "'best_random_forest_model.joblib'."
        )
        st.stop()

    st.title("📶 Dự đoán chất lượng dịch vụ mạng 4G")
    st.caption("Sử dụng mô hình Random Forest huấn luyện trên dữ liệu thông số mạng di động")

    st.markdown("### Nhập thông số mạng đo được")

    # --- Dùng st.form để gom các slider lại, tránh rerun (re-render DOM)
    # liên tục mỗi lần kéo slider -> đây là nguyên nhân chính gây lỗi
    # "removeChild" do React của Streamlit bị lệch node khi rerun quá nhanh.
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)

        with col1:
            rsrp = st.slider("RSRP (dBm)", min_value=-140, max_value=-40, value=-90,
                              help="Cường độ tín hiệu tham chiếu. Càng gần 0 càng tốt.",
                              key="rsrp_slider")
            rsrq = st.slider("RSRQ (dB)", min_value=-25, max_value=0, value=-12,
                              help="Chất lượng tín hiệu tham chiếu.",
                              key="rsrq_slider")
            sinr = st.slider("SINR (dB)", min_value=-10, max_value=35, value=12,
                              help="Tỷ lệ tín hiệu trên nhiễu. Càng cao càng tốt.",
                              key="sinr_slider")

        with col2:
            ping = st.slider("Ping (ms)", min_value=0, max_value=300, value=40,
                              help="Độ trễ mạng. Càng thấp càng tốt.",
                              key="ping_slider")
            download = st.slider("Download (Mbps)", min_value=0.0, max_value=200.0, value=25.0,
                                  help="Tốc độ tải xuống.",
                                  key="download_slider")
            upload = st.slider("Upload (Mbps)", min_value=0.0, max_value=100.0, value=8.0,
                                help="Tốc độ tải lên.",
                                key="upload_slider")

        submitted = st.form_submit_button(
            "🔍 Dự đoán chất lượng mạng", type="primary", use_container_width=True
        )

    if submitted:
        new_data = pd.DataFrame([{
            "RSRP_dBm": rsrp,
            "RSRQ_dB": rsrq,
            "SINR_dB": sinr,
            "Ping_ms": ping,
            "Download_Mbps": download,
            "Upload_Mbps": upload,
        }])[feature_cols]

        prediction = model.predict(new_data)[0]
        probabilities = model.predict_proba(new_data)[0]
        classes = model.classes_

        label_vn, emoji = QUALITY_LABELS.get(prediction, (prediction, ""))

        st.markdown("---")
        st.markdown(f"## Kết quả: {emoji} **{label_vn}** (`{prediction}`)")

        prob_df = pd.DataFrame({
            "Mức chất lượng": [QUALITY_LABELS.get(c, (c, ""))[0] for c in classes],
            "Xác suất (%)": (probabilities * 100).round(2),
        }).sort_values("Xác suất (%)", ascending=False)

        st.markdown("### Độ tin cậy của mô hình theo từng mức")
        st.dataframe(prob_df, hide_index=True, use_container_width=True)

        fig, ax = plt.subplots(figsize=(6, 3))
        ax.bar(prob_df["Mức chất lượng"], prob_df["Xác suất (%)"], color="steelblue")
        ax.set_ylabel("Xác suất (%)")
        ax.set_title("Phân bố xác suất dự đoán")
        st.pyplot(fig)
        plt.close(fig)  # đóng figure sau khi vẽ, tránh tích tụ object cũ giữa các lần rerun

    st.markdown("---")

    with st.expander("ℹ️ Mức độ ảnh hưởng của từng thông số (Feature Importance)"):
        importances = model.feature_importances_
        imp_df = pd.DataFrame({
            "Thông số": feature_cols,
            "Mức ảnh hưởng": importances
        }).sort_values("Mức ảnh hưởng", ascending=False)

        fig2, ax2 = plt.subplots(figsize=(6, 3))
        ax2.barh(imp_df["Thông số"], imp_df["Mức ảnh hưởng"], color="darkorange")
        ax2.invert_yaxis()
        st.pyplot(fig2)
        plt.close(fig2)  # đóng figure sau khi vẽ

        st.caption(
            "Cho biết thông số nào ảnh hưởng nhiều nhất đến quyết định "
            "phân loại chất lượng mạng của mô hình Random Forest."
        )

# ==============================================================
# TRANG 2: GIỚI THIỆU & HƯỚNG DẪN ĐO CHỈ SỐ
# ==============================================================
else:
    st.title("ℹ️ Giới thiệu & Hướng dẫn đo chỉ số mạng")

    st.markdown(
        """
        Ứng dụng này dự đoán **chất lượng dịch vụ mạng 4G** dựa trên 6 thông số kỹ thuật:
        `RSRP`, `RSRQ`, `SINR`, `Ping`, `Download`, `Upload`.

        Để lấy các chỉ số này trên điện thoại thực tế trước khi nhập vào ứng dụng,
        bạn có thể dùng một số app đo sóng / tốc độ mạng dưới đây.
        """
    )

    st.markdown("### 📡 Đo RSRP / RSRQ / SINR (chỉ số sóng di động)")
    st.markdown(
        """
        - **Network Cell Info Lite** (Android) – hiển thị đầy đủ RSRP, RSRQ, SINR, băng tần, cell ID theo thời gian thực.
        - **NetMonster** (Android) – tương tự Network Cell Info, giao diện hiện đại, hỗ trợ nhiều loại mạng (2G/3G/4G/5G).
        - **Signal Check Pro** (Android) – gọn nhẹ, tập trung vào tín hiệu và cường độ sóng.
        - **CellMapper** (Android/iOS, web) – đo sóng kết hợp bản đồ vị trí trạm phát sóng.

        > Lưu ý: trên iOS, Apple hạn chế quyền truy cập chi tiết mức sóng (RSRP/SINR),
        > nên các chỉ số này chủ yếu đo được đầy đủ trên **Android**. Với iPhone, có thể dùng
        > mã kỹ sư `*3001#12345#*` để xem một phần thông tin sóng.
        """
    )

    st.markdown("### ⏱️ Đo Ping / Download / Upload (tốc độ mạng)")
    st.markdown(
        """
        - **Speedtest by Ookla** (Android/iOS) – phổ biến nhất, đo Ping, Download, Upload, Jitter.
        - **nPerf** (Android/iOS) – đo tốc độ mạng kèm đánh giá chất lượng streaming, gọi video.
        - **Fast.com** (trình duyệt, Netflix) – đo nhanh tốc độ Download, đơn giản không cần cài app.
        - **Meteor** (Android, do Ookla phát triển) – đo thêm độ trễ khi chơi game, gọi video.
        """
    )

    st.markdown("### 🧭 Cách sử dụng cùng ứng dụng này")
    st.markdown(
        """
        1. Mở app đo sóng (vd. Network Cell Info Lite) để lấy **RSRP, RSRQ, SINR**.
        2. Mở app đo tốc độ (vd. Speedtest) để lấy **Ping, Download, Upload**.
        3. Quay lại trang **"Dự đoán chất lượng mạng"**, nhập các giá trị vừa đo được.
        4. Nhấn nút **Dự đoán** để xem kết quả và độ tin cậy của mô hình.
        """
    )

    st.info(
        "Các chỉ số đo được có thể dao động theo vị trí, thời điểm và thiết bị. "
        "Nên đo 2–3 lần và lấy giá trị trung bình để có kết quả ổn định hơn."
    )