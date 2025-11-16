"""
Silver Umbrella - Hybrid Explanation System
"""

import openai
from config import Config

# Initialize OpenAI
openai.api_key = Config.OPENAI_API_KEY

class ExplanationGenerator:
    """Generate senior-friendly explanations"""
    
    def __init__(self):
        self.high_confidence_threshold = Config.HIGH_CONFIDENCE_THRESHOLD
        self.low_confidence_threshold = Config.LOW_CONFIDENCE_THRESHOLD
    
    def generate_explanation(self, url, prediction, confidence, features):
        """Generate explanation based on confidence level"""
        
        if confidence >= self.high_confidence_threshold:
            return self._generate_template_explanation(url, prediction, confidence, features)
        else:
            return self._generate_ai_explanation(url, prediction, confidence, features)
    
    def _generate_template_explanation(self, url, prediction, confidence, features):
        """Generate rule-based explanation (FREE)"""
        
        is_phishing = (prediction == 0)
        
        red_flags = []
        
        if features.get('has_ip', 0) == 1:
            red_flags.append("Uses an IP address instead of a proper domain name")
        
        if features.get('num_hyphens', 0) > 3:
            red_flags.append(f"Contains {features['num_hyphens']} hyphens (unusual for legitimate sites)")
        
        if features.get('url_length', 0) > 75:
            red_flags.append(f"Very long URL ({features['url_length']} characters)")
        
        if features.get('subdomain_count', 0) > 3:
            red_flags.append(f"Too many subdomains ({features['subdomain_count']})")
        
        if features.get('num_at', 0) > 0:
            red_flags.append("Contains '@' symbol (often used in phishing)")
        
        if features.get('is_https', 0) == 0:
            red_flags.append("Not using secure HTTPS protocol")
        
        if features.get('num_dots', 0) > 5:
            red_flags.append(f"Excessive dots in URL ({features['num_dots']})")
        
        green_flags = []
        
        if features.get('is_https', 0) == 1:
            green_flags.append("Uses secure HTTPS connection")
        
        if features.get('url_length', 0) < 30:
            green_flags.append("Short, simple URL")
        
        if features.get('num_hyphens', 0) == 0:
            green_flags.append("No suspicious hyphens")
        
        if features.get('subdomain_count', 0) <= 1:
            green_flags.append("Normal domain structure")
        
        if is_phishing:
            explanation = {
                'verdict': 'DANGER - LIKELY PHISHING',
                'emoji': '⚠️',
                'color': 'red',
                'confidence_percent': int(confidence * 100),
                'summary': f"This URL shows {len(red_flags)} suspicious indicators commonly found in phishing attempts.",
                'details': red_flags if red_flags else ["Multiple suspicious patterns detected"],
                'advice': "DO NOT click this link. Do not enter any personal information. Delete any messages containing this link.",
                'explanation_type': 'template',
                'senior_friendly_summary': "🚨 STOP! This link is dangerous and trying to steal your information. Delete it immediately and do not click it."
            }
        else:
            explanation = {
                'verdict': 'SAFE - LEGITIMATE SITE',
                'emoji': '✅',
                'color': 'green',
                'confidence_percent': int(confidence * 100),
                'summary': f"This URL appears legitimate with {len(green_flags)} positive indicators.",
                'details': green_flags if green_flags else ["No major suspicious patterns detected"],
                'advice': "This site appears safe, but always verify you're visiting the correct website before entering personal information.",
                'explanation_type': 'template',
                'senior_friendly_summary': "✅ This link looks safe. However, always double-check you're on the right website before entering passwords or personal details."
            }
        
        return explanation
    
    def _generate_ai_explanation(self, url, prediction, confidence, features):
        """Generate AI-powered explanation using GPT-3.5-turbo"""
        
        is_phishing = (prediction == 0)
        feature_summary = self._summarize_features(features)
        
        prompt = f"""You are a cybersecurity expert explaining to a senior citizen (elderly person) whether a URL is safe or dangerous.

URL: {url}
Classification: {"PHISHING (dangerous)" if is_phishing else "LEGITIMATE (safe)"}
Confidence: {confidence*100:.1f}%

Technical indicators found:
{feature_summary}

Please provide:
1. A clear VERDICT (one sentence, very simple language)
2. WHY this URL is {"dangerous" if is_phishing else "safe"} (2-3 simple reasons)
3. WHAT ACTION to take (specific, clear instructions)

Use:
- Very simple language (5th grade reading level)
- Short sentences
- No technical jargon
- Friendly, protective tone
- Clear action steps

Keep response under 150 words."""

        try:
            response = openai.ChatCompletion.create(
                model=Config.OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": "You are a patient, kind cybersecurity expert who explains internet safety to seniors in very simple terms."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=200,
                temperature=0.7
            )
            
            ai_explanation = response.choices[0].message.content.strip()
            
            return {
                'verdict': 'DANGER - LIKELY PHISHING' if is_phishing else 'SAFE - LEGITIMATE SITE',
                'emoji': '⚠️' if is_phishing else '✅',
                'color': 'red' if is_phishing else 'green',
                'confidence_percent': int(confidence * 100),
                'summary': ai_explanation,
                'details': [],
                'advice': '',
                'explanation_type': 'ai',
                'senior_friendly_summary': ai_explanation
            }
            
        except Exception as e:
            print(f"AI explanation failed: {e}")
            return self._generate_template_explanation(url, prediction, confidence, features)
    
    def _summarize_features(self, features):
        """Create human-readable feature summary"""
        summary_parts = []
        
        if features.get('has_ip', 0) == 1:
            summary_parts.append("- Contains IP address")
        
        if features.get('num_hyphens', 0) > 2:
            summary_parts.append(f"- {features['num_hyphens']} hyphens in URL")
        
        if features.get('url_length', 0) > 60:
            summary_parts.append(f"- Long URL ({features['url_length']} characters)")
        
        if features.get('subdomain_count', 0) > 2:
            summary_parts.append(f"- {features['subdomain_count']} subdomains")
        
        if features.get('is_https', 0) == 1:
            summary_parts.append("- Uses HTTPS (secure)")
        else:
            summary_parts.append("- No HTTPS (insecure)")
        
        if not summary_parts:
            summary_parts.append("- Standard URL structure")
        
        return '\n'.join(summary_parts)

# Global instance
explainer = ExplanationGenerator()