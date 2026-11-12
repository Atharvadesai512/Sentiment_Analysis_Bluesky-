
import requests
import pandas as pd
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

URL = "https://api.bsky.app/xrpc/app.bsky.feed.searchPosts"

DAYS = 14
WINDOW_MINUTES = 15

# Use complete UTC days, excluding today
end = datetime.now(timezone.utc).replace(
    hour=0, minute=0, second=0, microsecond=0
)
start = end - timedelta(days=DAYS)

output = Path("data")
output.mkdir(exist_ok=True)

posts = {}
coverage = []

session = requests.Session()

current = start

while current < end:
    window_end = min(
        current + timedelta(minutes=WINDOW_MINUTES),
        end
    )

    params = {
        "q": "AI",
        "lang": "en",
        "sort": "latest",
        "limit": 100,
        "since": current.isoformat(),
        "until": window_end.isoformat()
    }

    success = False
    returned = 0
    valid = 0
    error = ""

    for attempt in range(3):
        try:
            response = session.get(
                URL, params=params, timeout=30
            )
            response.raise_for_status()

            result = response.json().get("posts", [])
            returned = len(result)

            for post in result:
                record = post.get("record", {})
                created = record.get("createdAt")

                if not created:
                    continue

                timestamp = datetime.fromisoformat(
                    created.replace("Z", "+00:00")
                )

                if not current <= timestamp < window_end:
                    continue

                uri = post.get("uri")
                if not uri:
                    continue

                valid += 1

                posts[uri] = {
                    "uri": uri,
                    "created_at": created,
                    "collected_at": datetime.now(
                        timezone.utc
                    ).isoformat(),
                    "text": record.get("text", ""),
                    "query": "AI"
                }

            success = True
            break

        except requests.RequestException as exc:
            error = str(exc)
            print("Retry:", current, error[:100])
            time.sleep(2 ** attempt)

    coverage.append({
        "window_start": current.isoformat(),
        "window_end": window_end.isoformat(),
        "success": success,
        "returned": returned,
        "valid": valid,
        "possibly_truncated": returned >= 100,
        "error": "" if success else error
    })

    print(
        current.strftime("%Y-%m-%d %H:%M"),
        "Valid:", valid,
        "Total unique:", len(posts)
    )

    current = window_end
    time.sleep(0.3)

pd.DataFrame(posts.values()).to_csv(
    output / "historical_posts.csv",
    index=False
)

pd.DataFrame(coverage).to_csv(
    output / "collection_coverage.csv",
    index=False
)

print("\nCollection finished")
print("Unique posts:", len(posts))
print("Windows:", len(coverage))
print("Failed windows:", sum(
    not x["success"] for x in coverage
))
print("Possibly truncated windows:", sum(
    x["possibly_truncated"] for x in coverage
))
