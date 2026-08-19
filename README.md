# VeriForward
A full-stack web application that detects misinformation and phishing links in forwarded WhatsApp and SMS messages using heuristic URL analysis and the Gemini API.

VeriForward is a web-based utility designed to counter spam, phishing, and fake news spread via forwarded text messages (e.g., on WhatsApp or SMS).

The application analyzes messages in two ways:

URL Safety (Heuristic Engine): Scrapes URLs inside the message and runs rules to detect brand spoofing, suspicious Top-Level Domains (TLDs), URL shorteners, missing HTTPS protocol, and IP-address-based hostnames.
Content Credibility (Gemini Engine): Uses the Gemini 3.5 Flash model to analyze the text claim for panic-inducing tone, high urgency, financial bait, or known misinformation tropes.
It runs as a Python FastAPI application (perfect for Vercel Serverless deployment) and has a clean, dark-mode, mobile-friendly React frontend.
