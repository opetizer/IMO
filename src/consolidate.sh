#!/bin/sh

TOPIC="MEPC"

echo "Starting data consolidation for topic: $TOPIC..."

python src/consolidate.py \
	--title "$TOPIC" \
&& \

echo "Consolidation for topic $TOPIC done!"

CSV_PATH="output/$TOPIC/trend_analysis.csv"
OUTPUT_PATH="output/$TOPIC/keyword_trends_dashboard.png"
CHART_TITLE_PREFIX="MEPC Keyword"
TOP_N=10

echo "Generating trend visualization dashboard..."
echo "Input CSV: $CSV_PATH"
echo "Top N per chart: $TOP_N"

python src/visualize_trends.py \
    --csv_path "output/MEPC/trend_analysis.csv" \
    --output_path "output/MEPC/keyword_trends.png" \
    --keywords ghg lng oil gas fuel carbon_intensity ghg_intensity ghg_reduction \
&& \

echo "Visualization done! Dashboard chart saved to $OUTPUT_PATH"
