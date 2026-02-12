#!/bin/sh

# =================================================================
# 用法: 
#   ./src/html_vis.sh [Session_Number] [Agenda_Items...]
# 
# 示例 1: 分析 MEPC 77 的所有议题
#   ./src/html_vis.sh 77
#
# 示例 2: 仅分析 MEPC 77 的议题 3 和 4
#   ./src/html_vis.sh 77 "3 4"
# =================================================================

TOPIC="MEPC"

# 检查是否传入了参数
if [ -z "$1" ]; then
  echo "Usage: $0 <Session_Suffix> [Agenda_Items_List]"
  echo "Example: $0 77 \"3 4\""
  exit 1
fi

if echo "$1" | grep -q "$TOPIC"; then
    SUBTITLE="$1"
else
    SUBTITLE="$TOPIC $1"
fi

AGENDA_ITEMS="$2"

echo "------------------------------------------------"
echo "Starting Visualization for: $SUBTITLE"

CMD="python src/html_vis.py --title \"$TOPIC\" --subtitle \"$SUBTITLE\""

# 如果传入了议题参数，则添加到命令中
if [ -n "$AGENDA_ITEMS" ]; then
    echo "Filtering for Agenda Items: $AGENDA_ITEMS"
    CMD="$CMD --agenda_items $AGENDA_ITEMS"
fi

echo "Executing: $CMD"
echo "------------------------------------------------"

# 执行命令
eval $CMD && \

echo "Visualization complete for $SUBTITLE!"
