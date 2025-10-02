# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Daily Decider is an AI-powered decision-making and compliment generator web application built with vanilla JavaScript, deployed on Netlify. It features advanced decision algorithms, sentiment analysis, pattern matching, and personalized daily compliments.

## Key Commands

### Development
- `npm run dev` - Start local development server on http://localhost:3000 (uses live-server)
- `npm test` - Run Jest tests (Note: No test files currently exist)

### Building & Deployment
- `npm run build` - Full build pipeline: optimizes assets, minifies code, copies public files to dist/
- `npm run optimize` - Run performance optimizations (minification, critical CSS, preload hints)
- `npm run deploy` - Deploy production build to Netlify (requires authentication)
- `npm run lighthouse` - Generate Lighthouse performance report

## Architecture

### Core JavaScript Modules (src/)
- **app.js**: Main application entry point, initializes engines and UI
  - Orchestrates DecisionEngine and ComplimentEngine
  - Handles form submissions and UI interactions
  - Manages analytics tracking

- **decision-engine.js**: Advanced decision-making algorithms
  - Multi-factor analysis with weighted scoring (temporal, pattern, sentiment, contextual)
  - Pattern recognition and user history tracking
  - Confidence scoring and alternative suggestions

- **compliment-engine.js**: Personalized compliment generation
  - 20 compliment categories with mood adaptation
  - Temporal awareness for time-appropriate messages
  - User preference learning

- **sentiment-analyzer.js**: Analyzes emotional context and mood
- **temporal-processor.js**: Time-based processing for contextual decisions
- **pattern-matcher.js**: Pattern recognition for decision optimization
- **analytics.js**: User behavior tracking and conversion funnel analytics
- **security.js**: Security utilities and input validation

### Build System
- **scripts/optimize.js**: Performance optimizer that creates production build
  - CSS/JS minification via Terser
  - Critical CSS generation and inlining
  - Resource preload hints
  - Service worker generation

### Deployment
- **netlify.toml**: Netlify configuration
  - Build settings with Node 18, Python 3.11
  - Security headers (CSP, XSS protection, etc.)
  - Serverless functions setup (directory: netlify/functions)
  - Caching strategies for static assets

### Data Flow
1. User input → app.js → DecisionEngine/ComplimentEngine
2. Engines use SentimentAnalyzer, TemporalProcessor, PatternMatcher
3. Analytics tracks user interactions
4. Results displayed with confidence scores and alternatives

## Important Notes

- The application uses vanilla JavaScript with no frontend framework
- All JS modules use ES6 classes and async/await patterns
- PayPal integration is via direct donation links (SDK removed due to issues)
- Frontend loads multiple engines that must be initialized in sequence
- Build output goes to dist/ directory for Netlify deployment
- No TypeScript - pure JavaScript throughout
- Analytics integration with custom dailyDeciderAnalytics global