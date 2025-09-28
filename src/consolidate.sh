#!/bin/sh

TOPIC="MEPC"

echo "Starting data consolidation for topic: $TOPIC..."

python src/consolidate.py \
	--title "$TOPIC" \
&& \

echo "Consolidation for $title done!"

CSV_PATH="output/$TOPIC/trend_analysis.csv"
OUTPUT_PATH="output/$TOPIC/keyword_trends.png"
CHART_TITLE="Keyword Trends for $TOPIC Sessions"
KEYWORDS="emission fuel ballast_water ghg carbon alternative_fuel energy_efficiency"

echo "Generating trend visualization..."
echo "Input CSV: $CSV_PATH"
echo "Keywords: $KEYWORDS"

python src/visualize_trends.py \
    --csv_path "$CSV_PATH" \
    --output_path "$OUTPUT_PATH" \
    --title "$CHART_TITLE" \
    --keywords $KEYWORDS \
&& \

echo "Visualization done! Chart saved to $OUTPUT_PATH"