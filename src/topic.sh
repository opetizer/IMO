#!/bin/sh

TOPIC="MEPC"

subtitle="$TOPIC$1"

OUTPUT_PATH="output/$TOPIC/$subtitle"

echo "Starting data consolidation for topic: $subtitle..."

python src/topic.py \
    --title "$TOPIC" \
    --subtitle "$subtitle" \
    --logging "log/logging.log" \
    --text_extracted_folder "$OUTPUT_PATH" \
    --num_topics $2 \
&& \

echo "Topic analysis for $subtitle done!"