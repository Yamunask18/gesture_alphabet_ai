import streamlit as st
import cv2
import time

from gesture_engine import GestureEngine


# -------------------------------
# PAGE CONFIG
# -------------------------------

st.set_page_config(
    page_title="Gesture AI",
    layout="wide"
)


# -------------------------------
# CUSTOM STYLE
# -------------------------------

st.markdown(
    """
    <style>

    .stApp {
        background-color: #0e1117;
        color: white;
    }


    h1, h2, h3, p, label {
        color: white !important;
    }


    .big-letter {

        font-size: 90px;
        text-align: center;
        font-weight: bold;
        color: white;

    }


    div.stButton > button {

        background-color: #2563eb;
        color: white;
        height: 45px;
        width: 200px;
        border-radius: 10px;
        border: none;
        font-size: 16px;
        font-weight: bold;

    }


    div.stButton > button:hover {

        background-color: #1d4ed8;
        color: white;

    }


    </style>
    """,
    unsafe_allow_html=True
)

# -------------------------------
# HEADER
# -------------------------------

st.title(
    "AI Gesture Recognition System"
)

st.caption(
    "Real Time Sign Gesture To Alphabet Conversion"
)


# -------------------------------
# LOAD ENGINE
# -------------------------------

@st.cache_resource
def load_engine():

    return GestureEngine()


engine = load_engine()



# -------------------------------
# SESSION STATE
# -------------------------------

if "running" not in st.session_state:

    st.session_state.running = False



# -------------------------------
# CONTROLS
# -------------------------------

c1, c2 = st.columns(2)


with c1:

    if st.button("Start Recognition"):

        st.session_state.running = True


with c2:

    if st.button("Stop Recognition"):

        st.session_state.running = False



# -------------------------------
# LAYOUT
# -------------------------------

left, right = st.columns(
    [1.5, 1]
)


with left:

    st.subheader(
        "Live Camera"
    )

    camera_box = st.empty()



with right:

    st.subheader(
        "Prediction Panel"
    )

    output_box = st.empty()



# -------------------------------
# CAMERA LOOP
# -------------------------------

if st.session_state.running:


    cap = cv2.VideoCapture(0)


    previous_time = time.time()


    while st.session_state.running:


        success, frame = cap.read()


        if not success:

            break



        frame = cv2.flip(
            frame,
            1
        )


        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )



        result = engine.process_frame(
            rgb
        )



        now = time.time()


        fps = int(
            1 /
            (
                now - previous_time
            )
        )


        previous_time = now



        small_frame = cv2.resize(
            rgb,
            (520, 360)
        )


        camera_box.image(
            small_frame,
            channels="RGB"
        )



        with output_box.container():


            if result["hand_detected"]:


                st.success(
                    "Hand Detected"
                )


            else:


                st.warning(
                    "Waiting For Gesture"
                )



            if result["prediction"]:


                st.markdown(
                    f"""
                    <div class='big-letter'>
                    {result["prediction"]}
                    </div>
                    """,
                    unsafe_allow_html=True
                )


                confidence = int(
                    result["confidence"] * 100
                )


                st.progress(
                    confidence
                )


                st.write(
                    "Confidence:",
                    str(confidence) + "%"
                )


            else:


                st.markdown(
                    """
                    <div class='big-letter'>
                    -
                    </div>
                    """,
                    unsafe_allow_html=True
                )



            st.metric(
                "FPS",
                fps
            )


            st.write(
                "Supported Gestures:"
            )


            st.write(
                "A  |  B  |  C  |  L  |  V"
            )



    cap.release()


else:


    camera_box.info(
        "Press Start Recognition"
    )