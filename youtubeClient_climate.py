import os
import sys
from googleapiclient.discovery import build


YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY", "ENTER_YOUR_YOUTUBE_API_KEY_HERE")


def youtubeClient():
    if YOUTUBE_API_KEY in ("ENTER_YOUR_YOUTUBE_API_KEY_HERE", ""):
        sys.exit("No YouTube API key set. Export YOUTUBE_API_KEY or edit this file.")

    client = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)
    print("  YouTube API client ready")
    return client
