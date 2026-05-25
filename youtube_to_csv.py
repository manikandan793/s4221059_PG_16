import json
import os
import sys
import pandas as pd


DATA_DIR   = "data"
INPUT_JSON = os.path.join(DATA_DIR, "youtubeDataDump.json")


def main():

    print("STEP 1 — YOUTUBE JSON to PIPELINE CSV")


    if not os.path.exists(INPUT_JSON):
        sys.exit(f"  Missing {INPUT_JSON}. Run fetchYoutubeData_climate.py first.")

    with open(INPUT_JSON, encoding="utf-8") as f:
        data = json.load(f)

    videos = data.get("videos", [])
    print(f"\n  Loaded {len(videos):,} videos from {INPUT_JSON}")

    # videos → posts
    posts_rows = []
    for v in videos:
        posts_rows.append({
            "post_id":      v["videoId"],
            "subreddit":    f"youtube_{v['camp']}",
            "camp":         v["camp"],
            "author":       v["channelTitle"],
            "title":        v.get("title", ""),
            "selftext":     v.get("description", ""),
            "score":        v.get("likeCount", 0),
            "num_comments": v.get("commentCount", 0),
            "created_utc":  v.get("publishedAt", ""),
            "url":          f"https://www.youtube.com/watch?v={v['videoId']}",
            "view_count":   v.get("viewCount", 0),
            "flair":        v.get("query", ""),
        })
    posts_df = pd.DataFrame(posts_rows)

    # comments → flat table with parent_author column for edges
    comments_rows = []
    for v in videos:
        video_id = v["videoId"]
        channel  = v["channelTitle"]
        camp     = v["camp"]
        for c in v.get("comments", []):
            author   = c.get("author", "[unknown]")
            reply_to = c.get("replyTo")
            parent   = reply_to if reply_to else channel

            comments_rows.append({
                "comment_id":    c.get("commentId", f"{video_id}_{author}"),
                "post_id":       video_id,
                "subreddit":     f"youtube_{camp}",
                "camp":          camp,
                "author":        author,
                "parent_id":     f"video_{video_id}" if reply_to is None else f"comment_{reply_to}",
                "parent_author": parent,
                "body":          c.get("text", ""),
                "score":         int(c.get("likeCount", 0)),
                "created_utc":   c.get("publishedAt", v.get("publishedAt", "")),
                "depth":         int(c.get("depth", 0)),
                "post_author":   channel,
            })
    comments_df = pd.DataFrame(comments_rows)

    posts_path    = os.path.join(DATA_DIR, "raw_posts.csv")
    comments_path = os.path.join(DATA_DIR, "raw_comments.csv")
    posts_df.to_csv(posts_path, index=False)
    comments_df.to_csv(comments_path, index=False)


    print(f"  Videos → posts    : {len(posts_df):,} rows → {posts_path}")
    print(f"  Comments          : {len(comments_df):,} rows → {comments_path}")
    print("\n  Posts by camp:")
    print(posts_df["camp"].value_counts().to_string())
    print("\n  Comments by camp:")
    print(comments_df["camp"].value_counts().to_string())
    print(f"\n  Unique commenters : {comments_df['author'].nunique():,}")
    print(f"  Unique channels   : {posts_df['author'].nunique():,}")



if __name__ == "__main__":
    main()
