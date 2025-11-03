"""Quick test to find what covenant language exists in the document."""
from src.ingestion.pdf_parser import PDFParser
from pathlib import Path
import re

def explore_document():
    pdf_path = Path('data/raw/SECExhibit 10.1 Term Loan Credit Agreement.pdf')
    parser = PDFParser(pdf_path)
    result = parser.extract()
    
    full_text = result['full_text']
    
    # Search for key covenant terms
    search_terms = [
        (r'Section\s+\d+\.\d+.*?[Pp]repayment', 'Prepayment'),
        (r'Section\s+\d+\.\d+.*?[Rr]edemption', 'Redemption'),
        (r'PIK|[Pp]ayment[- ]in[- ][Kk]ind', 'PIK'),
        (r'[Mm]andatory [Pp]repayment', 'Mandatory Prepayment'),
        (r'[Oo]ptional [Pp]repayment', 'Optional Prepayment'),
        (r'[Mm]ake[- ][Ww]hole', 'Make-Whole'),
        (r'[Cc]all [Pp]rotection', 'Call Protection'),
        (r'[Ee]xcess [Cc]ash [Ff]low', 'Excess Cash Flow'),
        (r'[Cc]ovenant', 'Covenant (general)'),
    ]
    
    print("🔍 Searching for covenant-related terms...\n")
    print("="*70)
    
    for pattern, name in search_terms:
        matches = list(re.finditer(pattern, full_text, re.IGNORECASE))
        print(f"\n📌 {name}: {len(matches)} matches")
        
        if matches:
            # Show first match with context
            match = matches[0]
            start = max(0, match.start() - 200)
            end = min(len(full_text), match.end() + 200)
            context = full_text[start:end]
            
            print(f"   First occurrence:")
            print(f"   {context[:300]}...")
            print()

if __name__ == '__main__':
    explore_document()