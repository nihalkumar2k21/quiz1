import pandas as pd
import json

def analyze_responses():
    """Analyze exam responses and flag suspicious behavior"""
    
    df = pd.read_csv('exam_responses.csv')
    
    print("=" * 60)
    print("EXAM RESPONSE ANALYSIS")
    print("=" * 60)
    print(f"\nTotal submissions: {len(df)}")
    print(f"Unique students: {df['roll_number'].nunique()}")
    
    # Flag suspicious behavior
    suspicious = df[
        (df['tab_switches'] > 3) | 
        (df['screenshot_attempts'] > 2) |
        (df['duration_seconds'] > 1300)  # >21.5 minutes
    ]
    
    if len(suspicious) > 0:
        print(f"\n⚠️ SUSPICIOUS ACTIVITY ({len(suspicious)} students):")
        print("-" * 60)
        for _, row in suspicious.iterrows():
            print(f"\nRoll: {row['roll_number']}")
            print(f"  Tab switches: {row['tab_switches']}")
            print(f"  Screenshot attempts: {row['screenshot_attempts']}")
            print(f"  Duration: {row['duration_seconds']/60:.1f} minutes")
    
    # Export for manual grading
    df.to_excel('exam_responses_for_grading.xlsx', index=False)
    print(f"\n✅ Exported to 'exam_responses_for_grading.xlsx'")
    
    # Show summary stats
    print("\n" + "=" * 60)
    print("VIOLATION SUMMARY:")
    print("=" * 60)
    print(f"Average tab switches: {df['tab_switches'].mean():.2f}")
    print(f"Average screenshot attempts: {df['screenshot_attempts'].mean():.2f}")
    print(f"Average duration: {df['duration_seconds'].mean()/60:.2f} minutes")

if __name__ == "__main__":
    analyze_responses()
