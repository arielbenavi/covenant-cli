"""
Extract text from PDF credit agreements with quality checks.
"""
import pdfplumber
from pathlib import Path
from typing import List, Dict
import re


class PDFParser:
    """Parse credit agreements and assess quality."""
    
    def __init__(self, pdf_path: str):
        self.pdf_path = Path(pdf_path)
        self.pages = []
        
    def extract(self) -> Dict:
        """
        Extract all text with page-level metadata.
        Returns: {
            'filename': str,
            'total_pages': int,
            'pages': [{'page_num': int, 'text': str, 'quality': float}],
            'full_text': str
        }
        """
        result = {
            'filename': self.pdf_path.name,
            'total_pages': 0,
            'pages': [],
            'full_text': ''
        }
        
        with pdfplumber.open(self.pdf_path) as pdf:
            result['total_pages'] = len(pdf.pages)
            
            for i, page in enumerate(pdf.pages):
                text = page.extract_text() or ""
                quality = self._assess_quality(text)
                
                page_data = {
                    'page_num': i + 1,
                    'text': text,
                    'quality': quality,
                    'char_count': len(text)
                }
                
                result['pages'].append(page_data)
                result['full_text'] += text + '\n\n'
        
        result['avg_quality'] = sum(p['quality'] for p in result['pages']) / len(result['pages']) if result['pages'] else 0
        return result
    
    def _assess_quality(self, text: str) -> float:
        """
        Heuristic quality score (0-1).
        High quality = mostly alphabetic, reasonable word length, has spaces.
        """
        if not text or len(text) < 50:
            return 0.0
        
        # Count alphabetic chars
        alpha_ratio = sum(c.isalpha() for c in text) / len(text)
        
        # Count spaces (should be ~15% of text)
        space_ratio = sum(c.isspace() for c in text) / len(text)
        space_score = 1.0 - abs(space_ratio - 0.15) * 2  # Penalize if too far from 15%
        space_score = max(0, space_score)
        
        # Check for common legal terms (sanity check)
        legal_terms = ['agreement', 'borrower', 'lender', 'covenant', 'section']
        term_score = sum(term in text.lower() for term in legal_terms) / len(legal_terms)
        
        return (alpha_ratio * 0.5 + space_score * 0.3 + term_score * 0.2)
    
    def extract_section(self, section_pattern: str) -> List[Dict]:
        """
        Extract specific sections (e.g., 'Section 2.0.*Prepayment').
        Returns matching text blocks with page numbers.
        """
        if not self.pages:
            self.extract()
        
        matches = []
        pattern = re.compile(section_pattern, re.IGNORECASE)
        
        for page in self.pages:
            if pattern.search(page['text']):
                # Find the actual matching text
                for match in pattern.finditer(page['text']):
                    # Extract surrounding context (500 chars before/after)
                    start = max(0, match.start() - 500)
                    end = min(len(page['text']), match.end() + 500)
                    
                    matches.append({
                        'page_num': page['page_num'],
                        'match_text': match.group(),
                        'context': page['text'][start:end]
                    })
        
        return matches


def test_parser():
    """Test on a local PDF file."""
    # Check which PDFs are available
    data_dir = Path('data/raw')
    
    if not data_dir.exists():
        print("❌ data/raw directory not found!")
        print("📁 Please create it and copy your PDFs there:")
        print("   mkdir -p data/raw")
        print("   cp ~/Downloads/*.pdf data/raw/")
        return
    
    pdf_files = list(data_dir.glob('*.pdf'))
    
    if not pdf_files:
        print("❌ No PDF files found in data/raw/")
        print("📁 Please copy your PDFs:")
        print("   cp ~/Downloads/SECExhibit_10_1_Term_Loan_Credit_Agreement.pdf data/raw/")
        return
    
    print(f"📂 Found {len(pdf_files)} PDF files:")
    for i, pdf in enumerate(pdf_files, 1):
        print(f"   {i}. {pdf.name}")
    
    # Test on the first PDF (credit agreement)
    test_pdf = None
    for pdf in pdf_files:
        if 'SECExhibit' in pdf.name or 'Credit_Agreement' in pdf.name:
            test_pdf = pdf
            break
    
    if not test_pdf:
        test_pdf = pdf_files[0]  # Just use first one
    
    print(f"\n🔬 Testing parser on: {test_pdf.name}")
    print("="*60)
    
    parser = PDFParser(test_pdf)
    result = parser.extract()
    
    print(f"\n📄 Filename: {result['filename']}")
    print(f"📊 Total Pages: {result['total_pages']}")
    print(f"✅ Average Quality: {result['avg_quality']:.2%}")
    print(f"📝 Total Characters: {len(result['full_text']):,}")
    
    # Show quality distribution
    print("\n📈 Quality by Page (first 5):")
    for page in result['pages'][:5]:
        quality_bar = "█" * int(page['quality'] * 20)
        print(f"  Page {page['page_num']:2d}: {page['quality']:.2%} {quality_bar} ({page['char_count']:,} chars)")
    
    # Show sample text
    print("\n📖 Sample text (first 500 chars):")
    print("-" * 60)
    print(result['full_text'][:500])
    print("-" * 60)
    
    # Test section extraction
    print("\n🔍 Testing Section Extraction (Prepayment):")
    matches = parser.extract_section(r'[Pp]repayment|[Rr]edemption')
    print(f"  Found {len(matches)} matches")
    if matches:
        print(f"\n  First match (Page {matches[0]['page_num']}):")
        print(f"  {matches[0]['match_text']}")
        print(f"\n  Context preview:")
        print(f"  {matches[0]['context'][:200]}...")
    
    return result


if __name__ == '__main__':
    test_parser()