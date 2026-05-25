import json
import os
import time
from youtubeClient_climate import youtubeClient


SCIENCE_QUERIES = [
    "climate change scientific evidence",
    "IPCC climate report explained",
    "global warming causes science",
    "climate change peer reviewed research",
    "greenhouse gas emissions explained",
    "climate scientists explain global warming",
    "renewable energy climate solution",
    "sea level rise evidence",
    "arctic ice melting climate change",
    "climate change 2024 evidence",
]

DENIAL_QUERIES = [
    "climate change hoax exposed",
    "global warming fake news",
    "climate change debunked",
    "climate scam conspiracy",
    "global warming not real",
    "climate change lie",
    "climate alarmism debunked",
    "climate scientists caught lying",
    "CO2 not causing warming",
    "climate change fraud exposed",
]

MAX_VIDEOS_PER_QUERY   = 50
MAX_COMMENTS_PER_VIDEO = 70
MAX_REPLIES_PER_THREAD = 10
OUTPUT_FILE            = "data/youtubeDataDump.json"


def search_videos(client, queries, camp):
    print(f"\n  Searching [{camp.upper()}] videos")
    seen, ids, meta = set(), [], {}

    for q in queries:
        print(f"    {q!r}")
        resp = client.search().list(
            q=q, part="snippet", type="video", order="viewCount",
            maxResults=min(MAX_VIDEOS_PER_QUERY, 50),
            relevanceLanguage="en", safeSearch="none",
        ).execute()

        for item in resp.get("items", []):
            vid = item["id"]["videoId"]
            if vid in seen:
                continue
            seen.add(vid)
            ids.append(vid)
            meta[vid] = {"snippet": item["snippet"], "camp": camp, "query": q}

        print(f"      unique videos so far: {len(ids)}")
        time.sleep(0.4)

    return ids, meta


def fetch_video_stats(client, video_ids):
    stats = {}
    for i in range(0, len(video_ids), 50):
        batch = video_ids[i:i + 50]
        resp = client.videos().list(
            id=",".join(batch),
            part="statistics,contentDetails",
        ).execute()
        for item in resp.get("items", []):
            stats[item["id"]] = item.get("statistics", {})
    return stats


def fetch_comments(client, video_id):
    resp = client.commentThreads().list(
        videoId=video_id, part="snippet,replies",
        maxResults=min(MAX_COMMENTS_PER_VIDEO, 100),
        textFormat="plainText", order="relevance",
    ).execute()

    comments = []
    for thread in resp.get("items", []):
        top      = thread["snippet"]["topLevelComment"]["snippet"]
        top_id   = thread["snippet"]["topLevelComment"]["id"]
        top_user = top["authorDisplayName"]

        comments.append({
            "commentId":   top_id,
            "author":      top_user,
            "replyTo":     None,
            "replyToType": "video",
            "text":        top["textDisplay"],
            "likeCount":   int(top.get("likeCount", 0)),
            "publishedAt": top["publishedAt"],
            "depth":       0,
        })

        replies = thread.get("replies", {}).get("comments", [])[:MAX_REPLIES_PER_THREAD]
        for reply in replies:
            r = reply["snippet"]
            comments.append({
                "commentId":   reply["id"],
                "author":      r["authorDisplayName"],
                "replyTo":     top_user,
                "replyToType": "comment",
                "text":        r["textDisplay"],
                "likeCount":   int(r.get("likeCount", 0)),
                "publishedAt": r["publishedAt"],
                "depth":       1,
            })

    return comments


def main():
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    client = youtubeClient()

    sci_ids, sci_meta = search_videos(client, SCIENCE_QUERIES, "science")
    den_ids, den_meta = search_videos(client, DENIAL_QUERIES,  "denial")
    all_ids  = sci_ids + den_ids
    all_meta = {**sci_meta, **den_meta}

    print(f"\n  Total videos: {len(all_ids)} (science={len(sci_ids)}, denial={len(den_ids)})")

    print("\n  Fetching video statistics")
    stats_map = fetch_video_stats(client, all_ids)

    print("\n  Fetching comments per video")
    videos = []
    for i, vid in enumerate(all_ids, 1):
        m       = all_meta[vid]
        snippet = m["snippet"]
        stats   = stats_map.get(vid, {})
        title   = snippet.get("title", "Unknown")

        comments = fetch_comments(client, vid)
        print(f"    [{i}/{len(all_ids)}] {title[:55]} → {len(comments)} comments")

        videos.append({
            "videoId":      vid,
            "camp":         m["camp"],
            "query":        m["query"],
            "title":        title,
            "channelTitle": snippet.get("channelTitle", ""),
            "publishedAt":  snippet.get("publishedAt", ""),
            "description":  snippet.get("description", "")[:500],
            "viewCount":    int(stats.get("viewCount", 0)),
            "likeCount":    int(stats.get("likeCount", 0)),
            "commentCount": int(stats.get("commentCount", 0)),
            "comments":     comments,
        })
        time.sleep(0.3)

    data = {
        "metadata": {
            "total_videos":    len(videos),
            "science_videos":  len(sci_ids),
            "denial_videos":   len(den_ids),
            "total_comments":  sum(len(v["comments"]) for v in videos),
            "science_queries": SCIENCE_QUERIES,
            "denial_queries":  DENIAL_QUERIES,
            "settings": {
                "max_videos_per_query":   MAX_VIDEOS_PER_QUERY,
                "max_comments_per_video": MAX_COMMENTS_PER_VIDEO,
                "max_replies_per_thread": MAX_REPLIES_PER_THREAD,
            },
        },
        "videos": videos,
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"\n  Saved {len(videos)} videos / {data['metadata']['total_comments']} comments → {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
