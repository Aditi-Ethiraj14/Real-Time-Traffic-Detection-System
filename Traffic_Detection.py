import streamlit as st
import cv2
import numpy as np
import tempfile
import time
import pandas as pd
from ultralytics import YOLO

# --- UI ---
st.set_page_config(page_title="Traffic Light & Vehicle Detection", layout="wide")
st.title("🚦 Traffic Light & Vehicle Detection")

st.markdown("""
This app detects **traffic lights (red, yellow, green)** and counts **vehicles only (cars, trucks, buses)**.
Upload a video file or use your webcam for live detection.
""")

# Sidebar options
source_option = st.sidebar.radio("Select input source:", ["Upload Video", "Webcam"])

# Traffic light HSV ranges
red_lower1, red_upper1 = np.array([0,150,100]), np.array([10,255,255])
red_lower2, red_upper2 = np.array([160,150,100]), np.array([179,255,255])
yellow_lower, yellow_upper = np.array([22,150,150]), np.array([33,255,255])
green_lower, green_upper = np.array([50,120,100]), np.array([70,255,255])

# Load YOLOv8 model for vehicles
vehicle_model = YOLO('yolov8n.pt')  # tiny model, detects vehicles

# --- Input Video ---
cap = None
frame_count = 0

if source_option == "Upload Video":
    uploaded_file = st.file_uploader("Upload Video", type=["mp4","jpg","jpeg","avi"])
    if uploaded_file:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        cap = cv2.VideoCapture(tfile.name)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
elif source_option == "Webcam":
    cap = cv2.VideoCapture(0)

# Collapse rows by time and values
def collapse_rows_time(df):
    collapsed = []
    prev_lights = None
    start_time = None
    vehicles_in_segment = []

    for _, row in df.iterrows():
        lights, vehicles, time_ = row["lights"], row["vehicles"], row["time"]

        if lights == prev_lights:
            # extend ongoing segment
            collapsed[-1]["end_time"] = time_
            vehicles_in_segment.append(vehicles)
            collapsed[-1]["vehicles"] = max(vehicles_in_segment)  # or np.mean(...) if you prefer avg
        else:
            # new segment begins
            vehicles_in_segment = [vehicles]
            collapsed.append({
                "start_time": time_,
                "end_time": time_,
                "lights": lights,
                "vehicles": vehicles
            })
        prev_lights = lights

    return pd.DataFrame(collapsed)


# Only proceed if cap is valid
if cap is not None:
    # Prepare CSV/report
    traffic_data = []
    vehicle_data = []

    stframe = st.empty()
    progress_bar = st.progress(0)
    frame_idx = 0

    # --- Detection function ---
    def detect_traffic_lights(frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        detections = []
        h, w = frame.shape[:2]
        roi = hsv[0:h, 0:w]

        def process_mask(mask, color_name):
            mask = cv2.GaussianBlur(mask, (7,7),0)
            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if 100 < area < 10000:
                    x,y,bw,bh = cv2.boundingRect(cnt)
                    aspect_ratio = bw/float(bh)
                    perimeter = cv2.arcLength(cnt, True)
                    if perimeter==0: continue
                    circularity = 4*np.pi*area/(perimeter**2)
                    if y > h*0.7: continue
                    if circularity>0.4 and 0.4<aspect_ratio<1.6 and bw<w*0.15 and bh<h*0.15:
                        detections.append((x,y,bw,bh,color_name))
        # Masks
        mask_red = cv2.inRange(roi, red_lower1, red_upper1) + cv2.inRange(roi, red_lower2, red_upper2)
        mask_yellow = cv2.inRange(roi, yellow_lower, yellow_upper)
        mask_green = cv2.inRange(roi, green_lower, green_upper)
        # Process
        process_mask(mask_red,"Red")
        process_mask(mask_yellow,"Yellow")
        process_mask(mask_green,"Green")
        return detections

    def augment_missing_lights(frame, tl_dets):
        # Look for Yellow and Red just above any detected Green in the same head.
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, w = frame.shape[:2]
        dets = tl_dets[:]  # copy

        def add_if_present(color, x, y, bw, bh):
            x0, y0 = max(0, x), max(0, y)
            x1, y1 = min(w, x + bw), min(h, y + bh)
            if x1 <= x0 or y1 <= y0:
                return

            roi = hsv[y0:y1, x0:x1]
            if color == "Yellow":
                mask = cv2.inRange(roi, yellow_lower, yellow_upper)
            elif color == "Red":
                mask = (cv2.inRange(roi, red_lower1, red_upper1) |
                        cv2.inRange(roi, red_lower2, red_upper2))
            else:
                return

            # If enough pixels match the color, accept it (tune 0.02 if needed)
            if cv2.countNonZero(mask) > 0.02 * (roi.shape[0] * roi.shape[1]):
                # avoid duplicates near the same spot
                for (xx, yy, bbw, bbh, cc) in dets:
                    if cc == color and abs(xx - x0) < bw and abs(yy - y0) < bh:
                        return
                dets.append((x0, y0, x1 - x0, y1 - y0, color))

        for (x, y, bw, bh, color) in tl_dets:
            if color != "Green":
                continue
            dy = int(1.25 * bh)  # estimated vertical spacing between bulbs
            # probe one bulb above for Yellow, two above for Red
            add_if_present("Yellow", x, y - dy, bw, bh)
            add_if_present("Red",    x, y - 2 * dy, bw, bh)

        return dets

    # --- Processing Loop ---
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx +=1
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

        # Traffic lights
        tl_dets = detect_traffic_lights(frame)
        tl_dets = augment_missing_lights(frame, tl_dets)
        detected_colors = [c for _,_,_,_,c in tl_dets]
        detected_text = ', '.join(detected_colors) if detected_colors else "None"

        # Vehicle detection
        results = vehicle_model.predict(frame, verbose=False)
        vehicles = [r for r in results[0].boxes if int(r.cls[0]) in [2,3,5,7]]  # car, motorcycle, bus, truck ids
        vehicle_count = len(vehicles)

        # Draw traffic lights
        for x,y,w_box,h_box,color in tl_dets:
            cv2.rectangle(frame,(x,y),(x+w_box,y+h_box),(0,255,0),2)
            cv2.putText(frame,color,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)

        # Draw vehicles
        for v in vehicles:
            x1,y1,x2,y2 = map(int,v.xyxy[0])
            cv2.rectangle(frame,(x1,y1),(x2,y2),(255,0,0),2)
            cv2.putText(frame,"Vehicle",(x1,y1-5),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,0,0),2)

        # Append data for report
        traffic_data.append({"frame":frame_idx,"time":timestamp,"lights":detected_text})
        vehicle_data.append({"frame":frame_idx,"time":timestamp,"vehicles":vehicle_count})

        # Display frame
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        stframe.image(frame_rgb, channels="RGB")
        if source_option=="Upload Video":
            progress_bar.progress(min(frame_idx/frame_count,1.0))

    # --- After Processing ---
    st.success("✅ Detection Complete!")

    # Consolidate CSV report
    traffic_df = pd.DataFrame(traffic_data)
    vehicle_df = pd.DataFrame(vehicle_data)
    report_df = traffic_df.merge(vehicle_df,on=["frame","time"])

    # Collapse by start/end time
    report_df = collapse_rows_time(report_df)

    st.subheader("Consolidated Detection Report")
    st.dataframe(report_df)

    # Download CSV
    report_csv = tempfile.NamedTemporaryFile(delete=False,suffix=".csv")
    report_df.to_csv(report_csv.name,index=False)
    st.download_button("Download CSV Report",report_csv.name,file_name="traffic_light_vehicle_report.csv")
