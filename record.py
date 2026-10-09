import subprocess
from datetime import datetime

# Dictionary of RTSP streams from config.py
CAMERAS = {
    "office_1": "rtsp://admin:aniket12@122.176.35.187:5554/Streaming/Channels/101",
    "office_2": "rtsp://admin:Cctv%401234@122.176.35.187:2554/cam/realmonitor?channel=02&subtype=0",
    "office_3": "rtsp://admin:Cctv%401234@122.176.35.187:2554/cam/realmonitor?channel=03&subtype=0",
    "office_4": "rtsp://admin:Admin%40123@122.176.35.187:554/cam/realmonitor?channel=3&subtype=0",
    "office_5": "rtsp://admin:Admin%40123@122.176.35.187:554/cam/realmonitor?channel=6&subtype=0"
}

# Duration in seconds (1 minute = 60 seconds)
DURATION = 60

def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    processes = []
    
    print(f"▶️ Starting recording for {len(CAMERAS)} cameras for {DURATION} seconds...")
    
    for cam_name, rtsp_url in CAMERAS.items():
        output_file = f"{cam_name}_{timestamp}.mp4"
        
        ffmpeg_cmd = [
            "ffmpeg",
            "-rtsp_transport", "tcp",
            "-i", rtsp_url,
            "-t", str(DURATION),
            "-c", "copy",
            "-an",
            output_file
        ]
        
        # Start the process non-blocking
        p = subprocess.Popen(ffmpeg_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        processes.append((cam_name, p, output_file))
        print(f"  [{cam_name}] Started recording -> {output_file}")
        
    print("⏳ Waiting for all recordings to finish...")
    
    # Wait for all processes to complete
    for cam_name, p, output_file in processes:
        p.wait()
        print(f"✅ [{cam_name}] Recording complete.")
        
    print("🎉 All recordings finished successfully.")

if __name__ == "__main__":
    main()