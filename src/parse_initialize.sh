#!/bin/sh

title="MEPC"
subtitle="MEPC$1"

mkdir -p "./output/$title/$subtitle"

echo "Running $title: $subtitle..."

# python src/parse_data.py \
#     --title "$title" \
#     --subtitle "$subtitle" \
#     --logging log/logging.log \
# && \

python src/cooccurrence.py \
    --title "$title" \
    --subtitle "$subtitle" \
    --logging log/logging.log \
    --text_extracted_folder "output/$title/$subtitle" \
    --countries "" \
    --start_date "" \
    --end_date "" \
&& \

echo "$title $subtitle done!"
