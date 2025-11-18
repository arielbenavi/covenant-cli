"""
Use LLM API to extract and score covenants from credit agreement text.
Supports both Claude (Anthropic).
"""
import anthropic
import requests
import os
from typing import List, Dict, Optional
import json
import yaml
from pathlib import Path
from dotenv import load_dotenv
import src

load_dotenv()


class CovenantExtractor:
    """
    Extract covenant clauses using LLM API.
    This creates our initial labeled dataset.
    """


    def __init__(self, provider: str = "claude", api_key: Optional[str] = None):
        """
        Initialize extractor.
        
        Args:
            provider: "claude"
            api_key: API key (or use env var ANTHROPIC_API_KEY)
        """
        self.provider = provider.lower()
        
        # Load configuration files
        self._load_config()
        
        if self.provider == "claude":
            self.client = anthropic.Anthropic(
                api_key=api_key or os.getenv('ANTHROPIC_API_KEY')
            )
        else:
            raise ValueError(f"Unknown provider: {provider}. Use 'claude'")
    
    def _load_config(self):
        """Load master prompt and weights from config files."""
        config_dir = Path(__file__).parent.parent.parent / 'config'
        
        # Load master prompt
        prompt_path = config_dir / 'master_prompt.txt'
        if not prompt_path.exists():
            raise FileNotFoundError(f"Master prompt not found: {prompt_path}")
        
        with open(prompt_path, 'r') as f:
            self.EXTRACTION_PROMPT = f.read()
        
        # Load weights
        weights_path = config_dir / 'covenant_weights.yaml'
        if not weights_path.exists():
            raise FileNotFoundError(f"Weights config not found: {weights_path}")
        
        with open(weights_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Store all config values
        self.dimension_weights = config['dimension_weights']
        self.beta = config['interaction_coefficient']
        self.SUB_WEIGHTS_EQUAL = config['sub_weights_equal']
        self.SUB_WEIGHTS_IMPORTANCE = config['sub_weights_importance']
        self.CRITICAL_COVENANTS = config['critical_covenants']
        self.coherence_params = config['coherence_params']
    
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
            else:
                print("no other provider available")
            
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
        """
        Calculate dimension scores using BOTH weighting schemes:
        1. Equal weights (baseline)
        2. Importance-based weights (hypothesis to test)
        
        Also includes the O×P interaction term (structural coherence).
        """
        covenant_details = {'E': {}, 'O': {}, 'P': {}}
        
        # Group found covenants
        for cov in covenants:
            dim = cov.get('dimension')
            cov_type = cov.get('covenant_type', 'unknown')
            score = cov.get('intensity_score', 3)
            
            if dim in covenant_details:
                covenant_details[dim][cov_type] = score
        
        # Calculate BOTH scoring methods
        scores_equal = self._calculate_dimension_scores(
            covenant_details, 
            self.SUB_WEIGHTS_EQUAL
        )
        
        scores_importance = self._calculate_dimension_scores(
            covenant_details, 
            self.SUB_WEIGHTS_IMPORTANCE
        )
        
        # Dimension scores with equal weights
        E_eq = scores_equal['E']
        O_eq = scores_equal['O']
        P_eq = scores_equal['P']
        
        # Dimension scores with importance weights
        E_imp = scores_importance['E']
        O_imp = scores_importance['O']
        P_imp = scores_importance['P']
        
        # Calculate O-P structural coherence (alignment between optionality and pacing)
        rho_OP = self._calculate_rho_OP(covenant_details['O'], covenant_details['P'])
        
        # Interaction coefficient (from paper: β adjusts how strongly coherence affects liquidity)
        beta = 0.15
        
        # Calculate base CLI (additive only)
        CLI_equal_base = (
            self.dimension_weights['E'] * E_eq + 
            self.dimension_weights['O'] * O_eq + 
            self.dimension_weights['P'] * P_eq
        )
        CLI_importance_base = (
            self.dimension_weights['E'] * E_imp + 
            self.dimension_weights['O'] * O_imp + 
            self.dimension_weights['P'] * P_imp
        )

        # Calculate interaction terms
        interaction_equal = self.beta * rho_OP * (O_eq * P_eq)
        interaction_importance = self.beta * rho_OP * (O_imp * P_imp)
        
        # Final CLI with interaction (full formula from paper)
        CLI_equal_full = CLI_equal_base + interaction_equal
        CLI_importance_full = CLI_importance_base + interaction_importance
        
        return {
            # Equal-weighted scores (baseline)
            'dimension_scores': {
                'E': round(E_eq, 2),
                'O': round(O_eq, 2),
                'P': round(P_eq, 2)
            },
            'CLI_base': round(CLI_equal_base, 2),  # Additive only
            'CLI': round(CLI_equal_full, 2),        # With interaction
            
            # Importance-weighted scores (hypothesis)
            'dimension_scores_importance': {
                'E': round(E_imp, 2),
                'O': round(O_imp, 2),
                'P': round(P_imp, 2)
            },
            'CLI_importance_base': round(CLI_importance_base, 2),  # Additive only
            'CLI_importance': round(CLI_importance_full, 2),        # With interaction
            
            # Interaction details
            'rho_OP': round(rho_OP, 2),
            'interaction_term_equal': round(interaction_equal, 2),
            'interaction_term_importance': round(interaction_importance, 2),
            'beta': beta,
            
            # Metadata
            'covenant_counts': {k: len(v) for k, v in covenant_details.items()},
            'covenant_details': covenant_details,
            'missing_critical_covenants': scores_equal['missing_critical']
        }

    def _calculate_rho_OP(self, O_covenants: Dict, P_covenants: Dict) -> float:
        """Calculate structural coherence between O and P."""
        alignment = 0.0
        params = self.coherence_params  # Use config values
        
        # Calculate averages
        O_avg = sum(O_covenants.values()) / len(O_covenants) if O_covenants else 1.0
        P_avg = sum(P_covenants.values()) / len(P_covenants) if P_covenants else 1.0
        
        # --- POSITIVE ALIGNMENT ---
        has_tight_pik = ('PIK_toggle' in O_covenants and 
                        O_covenants['PIK_toggle'] >= params['tight_pik_threshold'])
        has_cash_sweep = 'mandatory_cash_sweep' in P_covenants
        has_maintenance = 'maintenance_covenants' in P_covenants
        
        if has_tight_pik and (has_cash_sweep or has_maintenance):
            alignment += params['alignment_bonus_tight_pik_with_pacing']
        
        # --- NEGATIVE ALIGNMENT ---
        
        # Pattern 1: Both dimensions weak (covenant-lite)
        if O_avg <= params['weak_dimension_threshold'] and P_avg <= params['weak_dimension_threshold']:
            alignment += params['penalty_both_weak']
            
            # Extra penalty if BOTH are very weak
            if O_avg <= params['very_weak_threshold'] and P_avg <= params['very_weak_threshold']:
                alignment += params['penalty_both_very_weak']
        
        # Pattern 2: Missing critical covenants in BOTH dimensions
        missing_O_critical = 'PIK_toggle' not in O_covenants
        missing_P_critical = (
            'mandatory_cash_sweep' not in P_covenants and 
            'maintenance_covenants' not in P_covenants
        )
        
        if missing_O_critical and missing_P_critical:
            alignment += params['penalty_missing_critical']
        
        # Pattern 3: Weak amendment thresholds
        has_weak_amend = 'amend_waiver' in O_covenants and O_covenants['amend_waiver'] <= 2
        if has_weak_amend:
            alignment += params['penalty_weak_amendment']
        
        # Pattern 4: Loose PIK + No sweep
        has_loose_pik = 'PIK_toggle' in O_covenants and O_covenants['PIK_toggle'] <= 2
        no_cash_sweep = 'mandatory_cash_sweep' not in P_covenants
        
        if has_loose_pik and no_cash_sweep:
            alignment += params['penalty_loose_pik_no_sweep']
        
        return max(-1.0, min(1.0, alignment))

    def _calculate_dimension_scores(
            self, 
            covenant_details: Dict, 
            sub_weights: Dict
        ) -> Dict:
            """
            Helper method to calculate dimension scores with a given weighting scheme.
            """
            dim_scores = {}
            missing_critical = {}
            
            for dim in ['E', 'O', 'P']:
                weighted_sum = 0.0
                total_weight = 0.0
                
                # 1. Add weighted scores for FOUND covenants
                for cov_type, score in covenant_details[dim].items():
                    normalized_type = self._normalize_covenant_type(cov_type)
                    weight = sub_weights[dim].get(normalized_type, 0.05)
                    
                    weighted_sum += score * weight
                    total_weight += weight
                
                # 2. Add penalties for MISSING critical covenants
                missing_in_dim = []
                for critical_cov in self.CRITICAL_COVENANTS[dim]:
                    normalized_found = [
                        self._normalize_covenant_type(c) 
                        for c in covenant_details[dim].keys()
                    ]
                    
                    if critical_cov not in normalized_found:
                        weight = sub_weights[dim].get(critical_cov, 0.125)
                        weighted_sum += 1.0 * weight  # Missing = score of 1
                        total_weight += weight
                        missing_in_dim.append(critical_cov)
                
                missing_critical[dim] = missing_in_dim
                
                # 3. Calculate final score
                dim_scores[dim] = weighted_sum / total_weight if total_weight > 0 else 1.0
            
            return {
                'E': dim_scores['E'],
                'O': dim_scores['O'],
                'P': dim_scores['P'],
                'missing_critical': missing_critical
            }
        
    def _normalize_covenant_type(self, cov_type: str) -> str:
        """Normalize covenant type names."""
        mappings = {
            'call_protection': 'call_protection',
            'make_whole': 'make_whole',
            'make-whole': 'make_whole',
            'makewhole': 'make_whole',
            'pik_toggle': 'PIK_toggle',
            'pik': 'PIK_toggle',
            'payment_in_kind': 'PIK_toggle',
            'cash_sweep': 'mandatory_cash_sweep',
            'mandatory_prepayment': 'mandatory_cash_sweep',
            'mandatory_cash_sweep': 'mandatory_cash_sweep',
            'ecf_sweep': 'mandatory_cash_sweep',
            'excess_cash_flow': 'mandatory_cash_sweep',
            'accordion_incremental': 'accordion',
            'accordion': 'accordion',
            'amend_waiver_thresholds': 'amend_waiver',
            'restricted_payments': 'restricted_payments',
            'debt_incurrence_tests': 'debt_incurrence',
            'maintenance_covenants': 'maintenance_covenants',
            'information_rights': 'information_rights',
            'cross_default_acceleration': 'cross_default',
            'prepayment_waterfall': 'prepayment_waterfall',
            'mandatory_redemption': 'mandatory_redemption',
            'intercreditor_release': 'intercreditor_release',
        }
        
        normalized = cov_type.lower().replace(' ', '_').replace('-', '_')
        return mappings.get(normalized, normalized)

def test_extractor(provider: str = "claude"):
    """Test covenant extraction on the credit agreement."""
    from src.ingestion.pdf_parser import PDFParser
    
    print(f"🤖 Using {provider.upper()} API")
    print("="*70)
    
    # Find the credit agreement
    pdf_path = Path('data/raw/Amended and Restated Credit Agreement among LegalApp Holdings.pdf')
    
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
    
    # Show EQUAL-weighted scores (baseline)
    print(f"\n{'='*70}")
    print(f"📈 EQUAL WEIGHTS (Baseline)")
    print(f"{'='*70}")
    
    print(f"\nDimension Scores:")
    for dim, score in result.get('dimension_scores', {}).items():
        tier = "🟢 Strong" if score >= 4 else "🟡 Moderate" if score >= 3 else "🔴 Weak"
        weight = "40%" if dim == "E" else "35%" if dim == "O" else "25%"
        print(f"  {dim} (weight={weight:>3}): {score}/5.0  {tier}")
    
    # CLI scores with interaction breakdown
    cli_base = result.get('CLI_base', 0)
    cli_full = result.get('CLI', 0)
    rho_OP = result.get('rho_OP', 0)
    interaction = result.get('interaction_term_equal', 0)
    beta = result.get('beta', 0.15)
    
    print(f"\nCLI Calculation:")
    print(f"  Base CLI (additive):    {cli_base:.2f}/5.0")
    print(f"    = 0.40×{result['dimension_scores']['E']:.2f} + 0.35×{result['dimension_scores']['O']:.2f} + 0.25×{result['dimension_scores']['P']:.2f}")
    print(f"\n  Interaction term:       {interaction:+.3f}")
    print(f"    = β × ρ(O,P) × (O × P)")
    print(f"    = {beta} × {rho_OP:+.2f} × ({result['dimension_scores']['O']:.2f} × {result['dimension_scores']['P']:.2f})")
    print(f"    = {beta} × {rho_OP:+.2f} × {result['dimension_scores']['O'] * result['dimension_scores']['P']:.2f}")
    
    print(f"\n🎯 Final CLI (w/ interaction): {cli_full:.2f}/5.0")
    tier_label = get_tier_label(cli_full)
    print(f"   {tier_label}")
    
    # Explain rho
    if rho_OP > 0.2:
        rho_explanation = "✅ Aligned: Flexibility reinforces pacing discipline"
    elif rho_OP < -0.2:
        rho_explanation = "⚠️  Misaligned: Flexibility undermines pacing discipline"
    else:
        rho_explanation = "➖ Independent: Dimensions operate separately"
    
    print(f"\n  ρ(O,P) = {rho_OP:+.2f}  {rho_explanation}")
    
    # Show IMPORTANCE-weighted scores (hypothesis)
    print(f"\n{'='*70}")
    print(f"📈 IMPORTANCE WEIGHTS (Hypothesis)")
    print(f"{'='*70}")
    
    print(f"\nDimension Scores:")
    for dim, score in result.get('dimension_scores_importance', {}).items():
        tier = "🟢 Strong" if score >= 4 else "🟡 Moderate" if score >= 3 else "🔴 Weak"
        weight = "40%" if dim == "E" else "35%" if dim == "O" else "25%"
        print(f"  {dim} (weight={weight:>3}): {score}/5.0  {tier}")
    
    cli_imp_base = result.get('CLI_importance_base', 0)
    cli_imp_full = result.get('CLI_importance', 0)
    interaction_imp = result.get('interaction_term_importance', 0)
    
    print(f"\nCLI Calculation:")
    print(f"  Base CLI (additive):    {cli_imp_base:.2f}/5.0")
    print(f"  Interaction term:       {interaction_imp:+.3f}")
    print(f"\n🎯 Final CLI (w/ interaction): {cli_imp_full:.2f}/5.0")
    tier_label_imp = get_tier_label(cli_imp_full)
    print(f"   {tier_label_imp}")
    
    # Compare both methods
    print(f"\n{'='*70}")
    print(f"📊 COMPARISON")
    print(f"{'='*70}")
    
    delta_base = cli_imp_base - cli_base
    delta_full = cli_imp_full - cli_full
    
    print(f"\nEqual vs Importance Weights:")
    print(f"  Base CLI:  {cli_base:.2f} → {cli_imp_base:.2f}  (Δ {delta_base:+.2f})")
    print(f"  Full CLI:  {cli_full:.2f} → {cli_imp_full:.2f}  (Δ {delta_full:+.2f})")
    
    print(f"\nBase vs Full (with interaction):")
    print(f"  Equal weights:      {cli_base:.2f} → {cli_full:.2f}  (Δ {cli_full - cli_base:+.2f})")
    print(f"  Importance weights: {cli_imp_base:.2f} → {cli_imp_full:.2f}  (Δ {cli_imp_full - cli_imp_base:+.2f})")
    
    # Show missing critical covenants
    missing = result.get('missing_critical_covenants', {})
    if any(missing.values()):
        print(f"\n{'='*70}")
        print(f"⚠️  MISSING CRITICAL COVENANTS")
        print(f"{'='*70}")
        for dim, cov_list in missing.items():
            if cov_list:
                dim_name = "Exit Engineering" if dim == "E" else "Optionality" if dim == "O" else "Cash-Flow Pacing"
                print(f"\n{dim} ({dim_name}):")
                for cov in cov_list:
                    print(f"  • {cov.replace('_', ' ').title()}")
    
    # Show sample covenants
    print(f"\n{'='*70}")
    print(f"📋 SAMPLE COVENANTS (first 3)")
    print(f"{'='*70}")
    
    for i, cov in enumerate(result['covenants'][:3], 1):
        dim_name = "Exit" if cov.get('dimension') == "E" else "Optionality" if cov.get('dimension') == "O" else "Pacing"
        print(f"\n{i}. {cov.get('covenant_type', 'unknown').upper().replace('_', ' ')}")
        print(f"   Dimension: {cov.get('dimension', '?')} ({dim_name})")
        print(f"   Score: {cov.get('intensity_score', '?')}/5")
        print(f"   Section: {cov.get('section_reference', 'N/A')}")
        print(f"   Rationale: {cov.get('intensity_rationale', 'N/A')}")
        
        # Show numeric values if present
        numeric = cov.get('numeric_values', {})
        if numeric and any(v is not None for v in numeric.values()):
            print(f"   Numeric values:", end="")
            for key, val in numeric.items():
                if val is not None:
                    if key == 'threshold' or key == 'amount':
                        print(f" {key}=${val:,.0f}", end="")
                    else:
                        print(f" {key}={val}", end="")
            print()
        
        print(f"   Text: {cov.get('text', '')[:150]}...")
    
    # Save results
    output_dir = Path('data/labeled')
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{result['filename']}.json"
    
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"💾 Full results saved to: {output_file}")
    print(f"✅ Extraction complete!")
    print(f"{'='*70}\n")
    
    return result


def get_tier_label(cli: float) -> str:
    """Helper function to get tier label from CLI score."""
    if cli >= 4:
        return "Tier A (Structurally Liquid)"
    elif cli >= 3:
        return "Tier B (Moderately Liquid)"
    elif cli >= 2:
        return "Tier C (Weak Structure)"
    else:
        return "Tier D (Covenant-Lite)"


# Module entry point (keep this at the very end of the file)
if __name__ == '__main__':
    import sys
    
    provider = sys.argv[1] if len(sys.argv) > 1 else "claude"
    
    if provider not in ["claude", "mistral"]:
        print("Usage: python covenant_extractor.py [claude|mistral]")
        sys.exit(1)
    
    test_extractor(provider=provider)