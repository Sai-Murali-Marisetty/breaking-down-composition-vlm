#!/bin/bash
# CPU Tasks While Waiting for GPU
# Run analysis, data prep, and CLIP experiments on CPU

echo "======================================================"
echo "CPU-BASED TASKS FOR VLM PROJECT"
echo "======================================================"

# 1. Verify all data is loaded
echo ""
echo "1. Verifying datasets..."
python verify_aro_data.py

# 2. Re-run CLIP baseline (to verify on cluster)
echo ""
echo "2. Running CLIP baseline on cluster..."
python baseline_simple.py <<EOF
1
EOF

# 3. Analyze results
echo ""
echo "3. Analyzing results..."
python analyze_results.py

# 4. Generate visualizations (if you add plotting)
echo ""
echo "4. Generating visualizations..."
# python create_visualizations.py  # if you create this

# 5. Prepare data summaries
echo ""
echo "5. Creating data summaries..."
python -c "
from load_winoground_local import WinogroundLocalLoader
from load_aro_updated import AROLoader

print('Loading Winoground...')
wg = WinogroundLocalLoader()
wg.load()
print(f'  Total examples: {len(wg)}')

print('Loading ARO...')
aro = AROLoader()
aro.load()
print('  ARO loaded successfully')

print('Data verification complete!')
"

echo ""
echo "======================================================"
echo "CPU TASKS COMPLETE!"
echo "======================================================"
echo ""
echo "Ready for GPU tasks when A100 becomes available"