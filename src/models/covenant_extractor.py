"""
Use LLM API to extract and score covenants from credit agreement text.
Supports both Claude (Anthropic) and Mistral.
"""
import anthropic
import requests
import os
from typing import List, Dict, Optional
import json
from pathlib import Path
from dotenv import load_dotenv
import src

load_dotenv()


class CovenantExtractor:
    """
    Extract covenant clauses using LLM API.
    This creates our initial labeled dataset.
    """
    
    EXTRACTION_PROMPT = """You are a credit lawyer analyzing a credit agreement. Extract ALL covenant clauses related to liquidity timing and control.

**DIMENSIONS TO EXTRACT:**

**Exit Engineering (E):** Covenants that determine WHEN and HOW capital can exit
- Call protection periods (no-call provisions)
- Make-whole provisions (prepayment premiums)
- Optional prepayment terms
- Mandatory prepayment triggers
- Change of control provisions
- Portability/assignment rights

**Optionality Structuring (O):** Covenants that allow flexibility in timing
- PIK (Payment-in-Kind) toggles
- Most Favored Nation (MFN) clauses
- Equity kickers/warrants
- Accordion features (capacity expansion)
- Extension options

**Cash-Flow Pacing (P):** Covenants that regulate cash flow timing
- Excess Cash Flow sweeps (mandatory prepayment %)
- NAV/LTV triggers
- Distribution restrictions/lock-ups
- Cash sweep percentages and frequency
- Financial maintenance covenants affecting distributions

**SCORING RULES (1-5 scale):**

Exit Engineering:
- 5: ≥3 years call protection + defined make-whole formula
- 4: 2-3 years call protection + some premium
- 3: 1-2 years protection
- 2: <1 year protection
- 1: No call protection

Optionality (balance between flexibility and control):
- 5: Ratio-gated PIK with ≥100bps step-up (tight control)
- 4: Some ratio gates or step-ups
- 3: Discretionary but with limits
- 2: Broad flexibility, few constraints
- 1: Unlimited optionality

Cash-Flow Pacing:
- 5: ≥75% ECF sweep, automatic/quarterly trigger
- 4: 50-75% sweep, semi-annual
- 3: 50% sweep, annual
- 2: <50% sweep or discretionary
- 1: No cash sweep provisions

**OUTPUT FORMAT:**
Return a JSON array with each covenant as an object. Use this exact structure:

[
  {{
    "dimension": "E",
    "covenant_type": "call_protection",
    "text": "exact text from agreement",
    "section_reference": "Section X.X",
    "intensity_score": 4,
    "intensity_rationale": "2 years call protection with premium"
  }}
]

**IMPORTANT:**
- Quote exact text from the agreement (don't paraphrase)
- Extract ALL relevant covenants (aim for 10-20 per document)
- Be precise with intensity scores - reference actual numbers in the text
- Return ONLY the JSON array, no other text

**CREDIT AGREEMENT TEXT:**
{contract_text}"""
    
    def __init__(self, provider: str = "claude", api_key: Optional[str] = None):
        """
        Initialize extractor.
        
        Args:
            provider: "claude" or "mistral"
            api_key: API key (or use env var ANTHROPIC_API_KEY or MISTRAL_API_KEY)
        """
        self.provider = provider.lower()
        
        if self.provider == "claude":
            self.client = anthropic.Anthropic(
                api_key=api_key or os.getenv('ANTHROPIC_API_KEY')
            )
        elif self.provider == "mistral":
            self.api_key = api_key or os.getenv('MISTRAL_API_KEY')
            if not self.api_key:
                raise ValueError("MISTRAL_API_KEY not found in environment")
        else:
            raise ValueError(f"Unknown provider: {provider}. Use 'claude' or 'mistral'")
    
    def extract_covenants(self, contract_text: str, filename: str = "unknown") -> Dict:
        """
        Extract covenants from contract text.
        
        Returns: {
            'filename': str,
            'covenants': [list of covenant dicts],
            'dimension_scores': {'E': float, 'O': float, 'P': float},
            'CLI': float
        }
        """
        # Truncate if too long (most models have 100k-200k context)
        max_chars = 150000
        if len(contract_text) > max_chars:
            # Take first 50k + last 100k (covenants usually in Articles 4-7)
            contract_text = contract_text[:50000] + "\n\n[...TRUNCATED...]\n\n" + contract_text[-100000:]
        
        prompt = self.EXTRACTION_PROMPT.format(contract_text=contract_text)
        
        try:
            if self.provider == "claude":
                response_text = self._call_claude(prompt)
            else:  # mistral
                response_text = self._call_mistral(prompt)
            
            # Parse JSON response
            covenants = self._parse_json_response(response_text)
            
            # Calculate scores
            result = {
                'filename': filename,
                'covenants': covenants,
                'raw_response': response_text,
                'provider': self.provider
            }
            
            result.update(self._calculate_scores(covenants))
            
            return result
            
        except Exception as e:
            print(f"❌ Extraction failed: {e}")
            import traceback
            traceback.print_exc()
            return {
                'filename': filename,
                'covenants': [],
                'error': str(e)
            }
    
    def _call_claude(self, prompt: str) -> str:
        """Call Claude API."""
        message = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=16000,
            temperature=0,  # Deterministic for extraction
            messages=[{"role": "user", "content": prompt}]
        )
        return message.content[0].text
    
    def _call_mistral(self, prompt: str) -> str:
        """Call Mistral API."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": "mistral-medium-latest",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": 16000
        }
        
        response = requests.post(
            "https://api.mistral.ai/v1/chat/completions",
            headers=headers,
            json=data
        )
        response.raise_for_status()
        
        return response.json()["choices"][0]["message"]["content"]
    
    def _parse_json_response(self, response_text: str) -> List[Dict]:
        """Extract JSON array from LLM response (handles markdown formatting)."""
        # Try to find JSON array
        json_start = response_text.find('[')
        json_end = response_text.rfind(']') + 1
        
        if json_start == -1 or json_end == 0:
            print("⚠️  No JSON array found in response")
            print(f"Response preview: {response_text[:500]}")
            return []
        
        json_str = response_text[json_start:json_end]
        
        try:
            covenants = json.loads(json_str)
            return covenants
        except json.JSONDecodeError as e:
            print(f"⚠️  JSON parse error: {e}")
            print(f"Attempted to parse: {json_str[:200]}...")
            return []
    
    def _calculate_scores(self, covenants: List[Dict]) -> Dict:
        """Calculate dimension scores and CLI."""
        dimensions = {'E': [], 'O': [], 'P': []}
        
        for cov in covenants:
            dim = cov.get('dimension')
            score = cov.get('intensity_score', 3)
            if dim in dimensions:
                dimensions[dim].append(score)
        
        # Average per dimension (default to 1.0 if missing = covenant-lite)
        E = sum(dimensions['E']) / len(dimensions['E']) if dimensions['E'] else 1.0
        O = sum(dimensions['O']) / len(dimensions['O']) if dimensions['O'] else 1.0
        P = sum(dimensions['P']) / len(dimensions['P']) if dimensions['P'] else 1.0
        
        # CLI = weighted average (from research paper)
        CLI = 0.40 * E + 0.35 * O + 0.25 * P
        
        return {
            'dimension_scores': {
                'E': round(E, 2), 
                'O': round(O, 2), 
                'P': round(P, 2)
            },
            'CLI': round(CLI, 2),
            'covenant_counts': {k: len(v) for k, v in dimensions.items()}
        }


def test_extractor(provider: str = "claude"):
    """Test covenant extraction on the credit agreement."""
    from src.ingestion.pdf_parser import PDFParser
    
    print(f"🤖 Using {provider.upper()} API")
    print("="*70)
    
    # Find the credit agreement
    pdf_path = Path('data/raw/SECExhibit 10.1 Term Loan Credit Agreement.pdf')
    
    if not pdf_path.exists():
        print(f"❌ File not found: {pdf_path}")
        return
    
    print(f"\n🔄 Step 1: Extracting text from PDF...")
    parser = PDFParser(pdf_path)
    pdf_data = parser.extract()
    print(f"✅ Extracted {len(pdf_data['full_text']):,} characters from {pdf_data['total_pages']} pages\n")
    
    print(f"🔄 Step 2: Extracting covenants with {provider.upper()}...")
    print("   (This may take 30-60 seconds...)\n")
    
    extractor = CovenantExtractor(provider=provider)
    result = extractor.extract_covenants(
        pdf_data['full_text'],
        filename=pdf_data['filename']
    )
    
    if 'error' in result:
        print(f"❌ Error: {result['error']}")
        return
    
    # Display results
    print(f"\n{'='*70}")
    print(f"📊 RESULTS for {result['filename']}")
    print(f"{'='*70}")
    print(f"Provider: {result.get('provider', 'unknown').upper()}")
    print(f"Total covenants found: {len(result['covenants'])}")
    print(f"Covenant counts by dimension: {result.get('covenant_counts', {})}")
    
    print(f"\n📈 Dimension Scores:")
    for dim, score in result.get('dimension_scores', {}).items():
        tier = "🟢 Strong" if score >= 4 else "🟡 Moderate" if score >= 3 else "🔴 Weak"
        print(f"  {dim} (weight={'40%' if dim=='E' else '35%' if dim=='O' else '25%'}): {score}/5.0  {tier}")
    
    cli = result.get('CLI', 0)
    tier_label = "Tier A (Structurally Liquid)" if cli >= 4 else \
                 "Tier B (Moderately Liquid)" if cli >= 3 else \
                 "Tier C (Weak Structure)" if cli >= 2 else \
                 "Tier D (Covenant-Lite)"
    
    print(f"\n🎯 CLI Score: {cli}/5.0  ({tier_label})")
    
    # Show sample covenants
    print(f"\n📋 Sample Covenants (first 3):")
    print("="*70)
    for i, cov in enumerate(result['covenants'][:3], 1):
        print(f"\n{i}. {cov.get('covenant_type', 'unknown').upper()} (Dimension: {cov.get('dimension', '?')})")
        print(f"   Score: {cov.get('intensity_score', '?')}/5")
        print(f"   Section: {cov.get('section_reference', 'N/A')}")
        print(f"   Rationale: {cov.get('intensity_rationale', 'N/A')}")
        print(f"   Text: {cov.get('text', '')[:200]}...")
    
    # Save results
    output_dir = Path('data/labeled')
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{result['filename']}.json"
    
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"\n💾 Full results saved to: {output_file}")
    print(f"\n✅ Extraction complete!")
    
    return result


if __name__ == '__main__':
    import sys
    
    # Allow command line arg: python covenant_extractor.py mistral
    provider = sys.argv[1] if len(sys.argv) > 1 else "claude"
    
    if provider not in ["claude", "mistral"]:
        print("Usage: python covenant_extractor.py [claude|mistral]")
        sys.exit(1)
    
    test_extractor(provider=provider)