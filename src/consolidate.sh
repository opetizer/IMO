#!/bin/sh

TOPIC="MEPC"

echo "Starting data consolidation for topic: $TOPIC..."

python src/consolidate.py \
	--title "$TOPIC" \
&& \

echo "Consolidation for topic $TOPIC done!"

CSV_PATH="output/$TOPIC/trend_analysis.csv"
OUTPUT_PATH="output/$TOPIC/keyword_trends.png"
CHART_TITLE_PREFIX="MEPC Keyword"
TOP_N=10

echo "Generating trend visualization dashboard..."
echo "Input CSV: $CSV_PATH"
echo "Top N per chart: $TOP_N"

python src/visualize_trends.py \
    --csv_path "$CSV_PATH" \
    --output_path "$OUTPUT_PATH" \
    --keywords ghg bwms cii wind_propulsion \
&& \

echo "Visualization done! Dashboard chart saved to $OUTPUT_PATH"
