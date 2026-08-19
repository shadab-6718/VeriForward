import unittest
from unittest.mock import patch
from fastapi import HTTPException
from typing import Dict, Any

from app.url_analyzer import (
    extract_urls_and_claim,
    get_subdomain_count_and_primary,
    levenshtein_distance,
    analyze_urls,
    analyze_single_url
)
from app.combiner import combine_results, get_verdict
from app.main import analyze_message, AnalyzeRequest
from app.config import config

class TestRiskChecker(unittest.TestCase):
    
    def test_url_extraction(self):
        # Text with single HTTP URL
        text1 = "Hey visit http://example.com for info"
        urls, claim = extract_urls_and_claim(text1)
        self.assertEqual(urls, ["http://example.com"])
        self.assertEqual(claim, "Hey visit for info")
        
        # Text with multiple URLs and trailing punctuation
        text2 = "Check this: https://google.com, and also www.whatsapp.com/test! Okay?"
        urls, claim = extract_urls_and_claim(text2)
        self.assertIn("https://google.com", urls)
        self.assertIn("www.whatsapp.com/test", urls)
        self.assertEqual(len(urls), 2)
        self.assertEqual(claim, "Check this: and also Okay?")
        
        # Text with no URLs
        text3 = "This is a benign message without links."
        urls, claim = extract_urls_and_claim(text3)
        self.assertEqual(urls, [])
        self.assertEqual(claim, text3)

    def test_subdomain_count_and_primary(self):
        # Simple domain
        sub, primary = get_subdomain_count_and_primary("example.com")
        self.assertEqual(sub, 0)
        self.assertEqual(primary, "example")
        
        # With www
        sub, primary = get_subdomain_count_and_primary("www.example.com")
        self.assertEqual(sub, 0)
        self.assertEqual(primary, "example")
        
        # With subdomains
        sub, primary = get_subdomain_count_and_primary("sub.domain.example.com")
        self.assertEqual(sub, 2)
        self.assertEqual(primary, "example")
        
        # Two-part suffix (e.g., .co.uk)
        sub, primary = get_subdomain_count_and_primary("example.co.uk")
        self.assertEqual(sub, 0)
        self.assertEqual(primary, "example")
        
        sub, primary = get_subdomain_count_and_primary("sub.example.co.uk")
        self.assertEqual(sub, 1)
        self.assertEqual(primary, "example")

    def test_levenshtein(self):
        self.assertEqual(levenshtein_distance("whatsapp", "whatsapp"), 0)
        self.assertEqual(levenshtein_distance("whatsap", "whatsapp"), 1)
        self.assertEqual(levenshtein_distance("watsap", "whatsapp"), 2)
        self.assertEqual(levenshtein_distance("google", "facebook"), 8)

    def test_url_analyzer_scoring(self):
        # Legitimate HTTPS URL
        score, flags = analyze_single_url("https://www.google.com")
        self.assertEqual(score, 0.0)
        self.assertEqual(flags, [])
        
        # HTTP only
        score, flags = analyze_single_url("http://google.com")
        self.assertIn("NO_HTTPS", flags)
        self.assertGreater(score, 0.0)
        
        # Suspicious TLD
        score, flags = analyze_single_url("https://gift.xyz")
        self.assertIn("SUSPICIOUS_TLD", flags)
        
        # Known shortener
        score, flags = analyze_single_url("https://bit.ly/xyz")
        self.assertIn("URL_SHORTENER", flags)
        
        # IP Address Host
        score, flags = analyze_single_url("http://192.168.1.1/index.html")
        self.assertIn("IP_ADDRESS_HOST", flags)
        
        # Brand spoofing (typo)
        score, flags = analyze_single_url("https://www.whatsap.com")
        self.assertIn("BRAND_SPOOFING", flags)
        
        # Brand spoofing (compound suffix)
        score, flags = analyze_single_url("https://whatsapp-gift-free.com")
        self.assertIn("BRAND_SPOOFING", flags)

    def test_combiner(self):
        # Test weighted average
        res = combine_results(
            url_analysis={"risk_score": 0.4, "flags": ["NO_HTTPS", "URL_SHORTENER"]},
            claim_analysis={"risk_score": 0.6, "flags": ["FINANCIAL_BAIT"]},
            has_urls=True,
            has_claim=True
        )
        self.assertEqual(res["combined_score"], 0.5)
        self.assertEqual(res["verdict"], "medium risk")
        
        # Test override rule (> 0.85)
        res_override = combine_results(
            url_analysis={"risk_score": 0.9, "flags": ["IP_ADDRESS_HOST", "BRAND_SPOOFING"]},
            claim_analysis={"risk_score": 0.3, "flags": []},
            has_urls=True,
            has_claim=True
        )
        self.assertEqual(res_override["combined_score"], 0.9)
        self.assertEqual(res_override["verdict"], "high risk")

        # Test URL-only check fallback
        res_url_only = combine_results(
            url_analysis={"risk_score": 0.7, "flags": []},
            claim_analysis={"risk_score": 0.0, "flags": [], "reasoning": "No claim"},
            has_urls=True,
            has_claim=False
        )
        self.assertEqual(res_url_only["combined_score"], 0.7)
        self.assertEqual(res_url_only["verdict"], "high risk")
        
        # Test Claim-only check fallback
        res_claim_only = combine_results(
            url_analysis={"risk_score": 0.0, "flags": []},
            claim_analysis={"risk_score": 0.2, "flags": [], "reasoning": "Benign"},
            has_urls=False,
            has_claim=True
        )
        self.assertEqual(res_claim_only["combined_score"], 0.2)
        self.assertEqual(res_claim_only["verdict"], "low risk")


class TestApiEndpoints(unittest.TestCase):
    
    def setUp(self):
        # Ensure API key is configured for tests, or restore it in tearDown
        self.old_api_key = config.GEMINI_API_KEY
        config.GEMINI_API_KEY = "test-mock-key"
        
    def tearDown(self):
        config.GEMINI_API_KEY = self.old_api_key

    def test_analyze_endpoint_empty_message(self):
        # Empty message check
        with self.assertRaises(HTTPException) as ctx:
            analyze_message(AnalyzeRequest(message="   "))
        self.assertEqual(ctx.exception.status_code, 400)
        self.assertIn("empty or whitespace only", ctx.exception.detail)

    def test_analyze_endpoint_url_only(self):
        # URL only (no claim text)
        req = AnalyzeRequest(message="https://google.com")
        res = analyze_message(req)
        
        self.assertEqual(res["url_analysis"]["risk_score"], 0.0)
        self.assertEqual(res["claim_analysis"]["risk_score"], 0.0)
        self.assertEqual(res["combined_score"], 0.0)
        self.assertEqual(res["verdict"], "low risk")

    @patch("app.main.analyze_claim")
    def test_analyze_endpoint_claim_only(self, mock_analyze_claim):
        # Mock Gemini API response
        mock_analyze_claim.return_value = {
            "risk_score": 0.75,
            "flags": ["PANIC_INDUCING"],
            "reasoning": "Panic inducing forwarded text."
        }
        
        req = AnalyzeRequest(message="Urgent health announcement!")
        res = analyze_message(req)
        
        # Verify URL details are default/empty
        self.assertEqual(res["url_analysis"]["risk_score"], 0.0)
        self.assertEqual(res["url_analysis"]["flags"], [])
        
        # Verify Claim details
        self.assertEqual(res["claim_analysis"]["risk_score"], 0.75)
        self.assertEqual(res["claim_analysis"]["flags"], ["PANIC_INDUCING"])
        self.assertEqual(res["combined_score"], 0.75)
        self.assertEqual(res["verdict"], "high risk")

    @patch("app.main.analyze_claim")
    def test_analyze_endpoint_both(self, mock_analyze_claim):
        # Mock claim analysis response
        mock_analyze_claim.return_value = {
            "risk_score": 0.8,
            "flags": ["FINANCIAL_BAIT"],
            "reasoning": "Spam claim."
        }
        
        # Message with HTTP (url risk) and text (claim risk)
        req = AnalyzeRequest(message="Win a prize! Check http://bit.ly/xyz")
        res = analyze_message(req)
        
        # URL risk: http://bit.ly/xyz -> http (NO_HTTPS), bit.ly (URL_SHORTENER)
        # Expected URL score is 0.15 + 0.20 = 0.35
        self.assertAlmostEqual(res["url_analysis"]["risk_score"], 0.35)
        self.assertEqual(res["claim_analysis"]["risk_score"], 0.8)
        
        # Combined score = (0.35 + 0.8) / 2 = 0.575
        self.assertAlmostEqual(res["combined_score"], 0.575)
        self.assertEqual(res["verdict"], "medium risk")

    @patch("app.main.analyze_claim")
    def test_analyze_endpoint_both_with_override(self, mock_analyze_claim):
        # Mock claim analysis response with high risk
        mock_analyze_claim.return_value = {
            "risk_score": 0.9,
            "flags": ["HEALTH_MISINFO"],
            "reasoning": "Fake medicine warning."
        }
        
        req = AnalyzeRequest(message="Warning! Take this medicine immediately! Check http://valid-site.com")
        res = analyze_message(req)
        
        # URL risk: http://valid-site.com -> NO_HTTPS (0.15)
        self.assertAlmostEqual(res["url_analysis"]["risk_score"], 0.15)
        self.assertEqual(res["claim_analysis"]["risk_score"], 0.9)
        
        # Override rule should trigger because claim score is 0.9 (> 0.85)
        # Combined score should override to max(0.15, 0.9) = 0.9
        self.assertEqual(res["combined_score"], 0.9)
        self.assertEqual(res["verdict"], "high risk")

    def test_analyze_endpoint_missing_api_key(self):
        # Clear API key to trigger error handling
        config.GEMINI_API_KEY = ""
        
        req = AnalyzeRequest(message="Some random claim text")
        with self.assertRaises(HTTPException) as ctx:
            analyze_message(req)
            
        self.assertEqual(ctx.exception.status_code, 500)
        self.assertIn("GEMINI_API_KEY is not set", ctx.exception.detail)

if __name__ == "__main__":
    unittest.main()
