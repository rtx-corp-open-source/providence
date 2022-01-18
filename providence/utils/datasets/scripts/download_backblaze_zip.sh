#!/bin/sh

# meant to facilitate a download a la
# curl -L -o data.zip https://f001.backblazeb2.com/file/Backblaze-Hard-Drive-Data/data_Q4_2019.zip
# the obove works in terminal
echo "executing curl -L -o $2 $1"
# curl -L -o $2 $1

# borrowed from Chrome, because the download works there
curl $1 \
  -H 'Connection: keep-alive' \
  -H 'sec-ch-ua: " Not;A Brand";v="99", "Google Chrome";v="91", "Chromium";v="91"' \
  -H 'sec-ch-ua-mobile: ?0' \
  -H 'Upgrade-Insecure-Requests: 1' \
  -H 'Accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9' \
  -H 'Sec-Fetch-Site: none' \
  -H 'Sec-Fetch-Mode: navigate' \
  -H 'Sec-Fetch-User: ?1' \
  -H 'Sec-Fetch-Dest: document' \
  -H 'Accept-Language: en-US,en;q=0.9,ko;q=0.8' \
  --noproxy \
  --compressed -o $2