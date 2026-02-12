#!/bin/sh

title="MEPC"

subtitle="$title $1"

mkdir -p "./output/$title/$subtitle"

echo "Running $title: $subtitle..."

# python src/parse_data.py \
#     --title "$title" \
#     --subtitle "$subtitle" \
#     --logging log/logging.log \
# && \

echo "Processing JSON data for $subtitle..."

python src/json_process.py \
    --title "$title" \
    --subtitle "$subtitle" \
&& \

echo "JSON processing for $subtitle done!" && \

echo "$title $subtitle done!"
