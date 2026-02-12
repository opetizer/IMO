#!/bin/sh

# 设置主题
TOPIC="MEPC"

subtitle="$TOPIC $1"

OUTPUT_PATH="output/$TOPIC/$subtitle"

echo "Running Co-occurrence analysis for $subtitle..."

# 调用 python 脚本
python src/cooccurrence.py \
    --title "$TOPIC" \
    --subtitle "$subtitle" \
    --logging "log/cooccurrence_$subtitle.log" \
    --text_extracted_folder "$OUTPUT_PATH" \
    --per_agenda \
    --per_agenda_topk 80 \
    --countries "" \
    --start_date "" \
    --end_date "" \
&& \

echo "Co-occurrence analysis for $subtitle done!"