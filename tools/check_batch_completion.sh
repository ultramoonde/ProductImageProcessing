#!/bin/bash
BATCH_DIR="batch_processing_20251006_161240"
FINAL_CSV="$BATCH_DIR/batch_Final.csv"

echo "Monitoring batch processing..."
echo "Directory: $BATCH_DIR"
echo ""

while true; do
    # Check if final CSV exists
    if [ -f "$FINAL_CSV" ]; then
        echo "✅ BATCH COMPLETE!"
        echo "Final CSV: $FINAL_CSV"
        
        # Count products
        PRODUCTS=$(tail -n +2 "$FINAL_CSV" | wc -l | tr -d ' ')
        echo "Total products processed: $PRODUCTS"
        
        # List image folders
        echo ""
        echo "Processed images:"
        ls -1 "$BATCH_DIR" | grep IMG
        
        break
    fi
    
    # Show current progress
    IMAGES_DONE=$(ls -1d "$BATCH_DIR"/IMG_* 2>/dev/null | wc -l | tr -d ' ')
    echo "[$(date +%H:%M:%S)] Processing... ($IMAGES_DONE/4 images started)"
    
    # Check for FINAL_RESULTS.csv files
    FINALS=$(find "$BATCH_DIR" -name "*FINAL_RESULTS.csv" 2>/dev/null | wc -l | tr -d ' ')
    echo "             Individual images completed: $FINALS/4"
    
    sleep 30
done
