
import requests
import time
from datetime import datetime, timedelta, timezone

url = "https://api.bsky.app/xrpc/app.bsky.feed.searchPosts"

start = datetime(2026, 10, 29, 0, 0, tzinfo=timezone.utc)

all_posts = {}
hourly_counts = []

for hour in range(24):
    hour_start = start + timedelta(hours=hour)
    hour_end = hour_start + timedelta(hours=1)

    params = {
        "q": "AI",
        "sort": "latest",
        "limit": 100,
        "lang": "en",
        "since": hour_start.isoformat(),
        "until": hour_end.isoformat()
    }

    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()

        posts = response.json().get("posts", [])
        valid = {}

        for post in posts:
            record = post.get("record", {})
            created = record.get("createdAt")

            if not created:
                continue

            timestamp = datetime.fromisoformat(
                created.replace("Z", "+00:00")
            )

            if hour_start <= timestamp < hour_end:
                valid[post["uri"]] = post
                all_posts[post["uri"]] = post

        hourly_counts.append(len(valid))

        print(
            hour_start.strftime("%Y-%m-%d %H:%M"),
            "Posts:", len(valid)
        )

    except requests.RequestException as e:
        hourly_counts.append(None)
        print("FAILED:", hour_start, str(e))

    time.sleep(1)

print("\nTotal unique posts:", len(all_posts))
print("Successful hours:", sum(x is not None for x in hourly_counts))
print("Failed hours:", sum(x is None for x in hourly_counts))
print("Hours with zero posts:", sum(x == 0 for x in hourly_counts))
print("Hours with 90+ posts:", sum(x is not None and x >= 90 for x in hourly_counts))
