#!/bin/bash
for f in annotated_*.mp4; do
    if [[ ! "$f" == *"h264"* ]]; then
        echo "Converting $f for browser compatibility..."
        ffmpeg -y -loglevel warning -i "$f" -vcodec libx264 -pix_fmt yuv420p "h264_$f"
    fi
done
echo "All done! The h264_ files are fully compatible with local_viewer.html"
