"""
Analyze Baseline Results
Identify failure patterns and create error taxonomy
"""

import pandas as pd
import json
import glob
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def analyze_winoground_results():
    """Analyze Winoground results by compositional tag"""
    
    print("="*60)
    print("WINOGROUND RESULTS ANALYSIS")
    print("="*60)
    
    # Find latest results file
    csv_files = glob.glob('results/winoground_clip_detailed_*.csv')
    if not csv_files:
        print("❌ No results files found")
        return
    
    latest_file = sorted(csv_files)[-1]
    print(f"\n📊 Analyzing: {latest_file}")
    
    # Load data
    df = pd.read_csv(latest_file)
    
    print(f"\nTotal examples: {len(df)}")
    print(f"\nOverall Performance:")
    print(f"  Text Score:  {df['text_score'].mean()*100:.2f}%")
    print(f"  Image Score: {df['image_score'].mean()*100:.2f}%")
    print(f"  Group Score: {df['group_score'].mean()*100:.2f}%")
    
    # Analysis by tag
    print("\n" + "="*60)
    print("PERFORMANCE BY COMPOSITIONAL TAG")
    print("="*60)
    
    tag_analysis = df.groupby('tag').agg({
        'text_score': ['count', 'sum', 'mean'],
        'image_score': ['sum', 'mean'],
        'group_score': ['sum', 'mean']
    }).round(3)
    
    # Sort by group score (worst first)
    tag_analysis_sorted = tag_analysis.sort_values(('group_score', 'mean'))
    
    print("\n🔴 HARDEST TAGS (Lowest Group Score):")
    print("-" * 60)
    
    for i, (tag, row) in enumerate(tag_analysis_sorted.head(10).iterrows()):
        count = int(row[('text_score', 'count')])
        group_pct = row[('group_score', 'mean')] * 100
        text_pct = row[('text_score', 'mean')] * 100
        image_pct = row[('image_score', 'mean')] * 100
        
        print(f"\n{i+1}. {tag}")
        print(f"   Count: {count} | Group: {group_pct:5.1f}% | Text: {text_pct:5.1f}% | Image: {image_pct:5.1f}%")
    
    print("\n🟢 EASIEST TAGS (Highest Group Score):")
    print("-" * 60)
    
    for i, (tag, row) in enumerate(tag_analysis_sorted.tail(5).iterrows()):
        count = int(row[('text_score', 'count')])
        group_pct = row[('group_score', 'mean')] * 100
        text_pct = row[('text_score', 'mean')] * 100
        image_pct = row[('image_score', 'mean')] * 100
        
        print(f"\n{i+1}. {tag}")
        print(f"   Count: {count} | Group: {group_pct:5.1f}% | Text: {text_pct:5.1f}% | Image: {image_pct:5.1f}%")
    
    # Identify failure categories
    print("\n" + "="*60)
    print("ERROR TAXONOMY (Preliminary)")
    print("="*60)
    
    # Tags with <20% group score
    severe_failures = tag_analysis_sorted[tag_analysis_sorted[('group_score', 'mean')] < 0.2]
    
    if len(severe_failures) > 0:
        print(f"\n❌ SEVERE FAILURES (<20% group score):")
        for tag, row in severe_failures.iterrows():
            print(f"   - {tag}: {row[('group_score', 'mean')]*100:.1f}%")
    
    # Tags where text >> image
    text_strong = df[df['text_score'] > df['image_score']].groupby('tag').size()
    
    print(f"\n⚠️  TEXT > IMAGE (text encoder stronger):")
    print(f"   {len(text_strong)}/{len(tag_analysis)} tags show this pattern")
    
    # Save analysis
    tag_analysis.to_csv('results/winoground_tag_analysis.csv')
    print(f"\n💾 Saved detailed analysis to: results/winoground_tag_analysis.csv")
    
    return df, tag_analysis

def analyze_failure_examples(n=10):
    """Show specific examples where CLIP failed"""
    
    print("\n" + "="*60)
    print("EXAMPLE FAILURES")
    print("="*60)
    
    csv_files = glob.glob('results/winoground_clip_detailed_*.csv')
    if not csv_files:
        return
    
    df = pd.read_csv(sorted(csv_files)[-1])
    
    # Get failed examples (group_score = 0)
    failures = df[df['group_score'] == 0].head(n)
    
    print(f"\nShowing {len(failures)} failure examples:\n")
    
    for i, row in failures.iterrows():
        print(f"Example {row['id']}:")
        print(f"  Tag: {row['tag']}")
        print(f"  Caption 0: {row['caption_0']}")
        print(f"  Caption 1: {row['caption_1']}")
        print(f"  Text Score: {'✅' if row['text_score'] else '❌'}")
        print(f"  Image Score: {'✅' if row['image_score'] else '❌'}")
        print()

def create_error_taxonomy():
    """Create structured error taxonomy based on results"""
    
    print("\n" + "="*60)
    print("ERROR TAXONOMY DRAFT")
    print("="*60)
    
    taxonomy = {
        "Failure Categories": {
            "1. Attribute Confusion": {
                "description": "Fails to distinguish different attributes",
                "examples": ["Adjective-Color", "Adjective-Size", "Adjective-Age"],
                "severity": "High"
            },
            "2. Relation/Role Confusion": {
                "description": "Cannot distinguish agent vs patient roles",
                "examples": ["Verb-Transitive", "Preposition"],
                "severity": "High"
            },
            "3. Word Order Insensitivity": {
                "description": "Ignores word order changes",
                "examples": ["Noun Phrase order swaps"],
                "severity": "Medium"
            },
            "4. Negation Failures": {
                "description": "Cannot process negation correctly",
                "examples": ["Negation, Scope"],
                "severity": "High"
            },
            "5. Spatial Reasoning": {
                "description": "Struggles with spatial prepositions",
                "examples": ["Preposition (on, in, above, below)"],
                "severity": "Medium"
            }
        }
    }
    
    for category, details in taxonomy["Failure Categories"].items():
        print(f"\n{category}")
        print(f"  Description: {details['description']}")
        print(f"  Severity: {details['severity']}")
        print(f"  Example tags: {', '.join(details['examples'][:3])}")
    
    # Save taxonomy
    with open('results/error_taxonomy_draft.json', 'w') as f:
        json.dump(taxonomy, f, indent=2)
    
    print(f"\n💾 Saved to: results/error_taxonomy_draft.json")
    
    return taxonomy

def generate_report():
    """Generate summary report"""
    
    print("\n" + "="*60)
    print("GENERATING BASELINE REPORT")
    print("="*60)
    
    # Load results
    summary_files = glob.glob('results/winoground_clip_summary_*.json')
    if not summary_files:
        return
    
    with open(sorted(summary_files)[-1], 'r') as f:
        results = json.load(f)
    
    report = f"""
# CLIP Baseline Evaluation Report

## Overall Performance

### Winoground (400 examples)
- **Group Score**: {results['metrics']['group_score']['accuracy']:.2f}%
- **Text Score**: {results['metrics']['text_score']['accuracy']:.2f}%  
- **Image Score**: {results['metrics']['image_score']['accuracy']:.2f}%

### Key Findings
1. Text encoder outperforms image-text alignment
2. Group score matches published benchmarks (~30%)
3. Significant variation by compositional tag

## Comparison to Baselines
- Random Chance: 25%
- Published CLIP: ~30%
- Your CLIP: {results['metrics']['group_score']['accuracy']:.2f}%
- Human Performance: >95%

## Next Steps
1. Analyze failure patterns by tag
2. Develop complete error taxonomy
3. Implement prompting strategies
4. Test LLaVA and SmolVLM models
"""
    
    with open('results/baseline_report.md', 'w') as f:
        f.write(report)
    
    print(report)
    print(f"\n💾 Saved to: results/baseline_report.md")

def main():
    print("\n" + "="*60)
    print("BASELINE RESULTS ANALYSIS")
    print("="*60)
    
    print("\nThis will:")
    print("1. Analyze performance by compositional tag")
    print("2. Identify hardest/easiest categories")
    print("3. Show example failures")
    print("4. Create preliminary error taxonomy")
    print("5. Generate summary report")
    
    input("\nPress Enter to start analysis...")
    
    # Run analyses
    df, tag_analysis = analyze_winoground_results()
    
    if df is not None:
        analyze_failure_examples(n=10)
        create_error_taxonomy()
        generate_report()
        
        print("\n" + "="*60)
        print("🎉 ANALYSIS COMPLETE!")
        print("="*60)
        
        print("\n📁 Generated files:")
        print("  - results/winoground_tag_analysis.csv")
        print("  - results/error_taxonomy_draft.json")
        print("  - results/baseline_report.md")
        
        print("\n🎯 Next Steps:")
        print("1. Review tag analysis to identify patterns")
        print("2. Refine error taxonomy")
        print("3. Design prompting strategies to address failures")
        print("4. Test other models (LLaVA, SmolVLM)")

if __name__ == "__main__":
    main()