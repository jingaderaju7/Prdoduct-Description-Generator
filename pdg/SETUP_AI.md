# AI Product Analyzer - Setup Guide

This project now supports AI-powered product detection and description generation using Vision APIs from Claude (Anthropic) or GPT-4 (OpenAI).

## Features

✨ **AI-Powered Analysis** - Automatically detects products from images with high accuracy
🤖 **Realistic Descriptions** - Generates natural, product-focused descriptions like ChatGPT or Gemini
📊 **Visual Analysis** - Provides detailed information about material, condition, style, and target audience
🎯 **Fallback Support** - Falls back to traditional CV if API is unavailable

## Setup Instructions

### Option 1: Using Claude API (Recommended)

1. **Get an API Key**
   - Visit [Anthropic Console](https://console.anthropic.com/)
   - Sign up for an account
   - Create an API key in your account settings

2. **Set Environment Variable**
   
   **Windows PowerShell:**
   ```powershell
   $env:ANTHROPIC_API_KEY="your-api-key-here"
   python app.py
   ```
   
   **Windows Command Prompt:**
   ```cmd
   set ANTHROPIC_API_KEY=your-api-key-here
   python app.py
   ```
   
   **Linux/Mac:**
   ```bash
   export ANTHROPIC_API_KEY="your-api-key-here"
   python app.py
   ```

3. **Permanent Setup (Windows)**
   - Open System Properties → Environment Variables
   - Click "New" under User variables
   - Variable name: `ANTHROPIC_API_KEY`
   - Variable value: `your-api-key-here`
   - Click OK and restart your terminal/IDE

### Option 2: Using OpenAI API (GPT-4 Vision)

1. **Get an API Key**
   - Visit [OpenAI API Keys](https://platform.openai.com/api-keys)
   - Create a new secret key

2. **Set Environment Variable**
   
   **Windows PowerShell:**
   ```powershell
   $env:OPENAI_API_KEY="your-api-key-here"
   python app.py
   ```
   
   **Windows Command Prompt:**
   ```cmd
   set OPENAI_API_KEY=your-api-key-here
   python app.py
   ```

## How It Works

1. **Image Upload**
   - User uploads a product image through the web interface
   - Image is saved locally

2. **AI Analysis** (if API key is set)
   - Image is encoded to base64
   - Sent to Claude or OpenAI Vision API
   - API analyzes the image and detects:
     - Product name
     - Category
     - Brand
     - Key features
     - Material, condition, style
     - Unique characteristics

3. **Description Generation**
   - If AI analysis succeeds: Uses AI-generated realistic description
   - If AI unavailable: Falls back to traditional ML/CV approach
   - Shows confidence score and analysis details

## API Costs

### Claude API (Anthropic)
- Vision images: ~$0.01 per analyzed image
- Much cheaper than alternatives
- Good balance of cost and accuracy

### OpenAI GPT-4 Vision
- Vision images: ~$0.01-0.03 per analyzed image
- Reliable and widely used
- High accuracy

## Testing Without API Key

The app works without an API key! It will automatically fallback to traditional computer vision analysis which:
- Uses edge detection for shape analysis
- Analyzes colors and textures
- Generates category predictions
- Extracts visual features

This provides good results, though not as realistic as AI-powered analysis.

## Troubleshooting

1. **"No module named 'anthropic'"**
   ```bash
   pip install anthropic
   ```

2. **API key not being recognized**
   - Make sure environment variable is set BEFORE running app
   - Restart terminal/IDE after setting environment variable
   - Check variable name is exactly: `ANTHROPIC_API_KEY` or `OPENAI_API_KEY`

3. **Rate limiting errors**
   - API keys have rate limits
   - Wait a few seconds before making next request
   - Check your API key quota on the provider's dashboard

4. **Invalid API key**
   - Double-check the API key is correct
   - No extra spaces or newlines
   - Make sure key hasn't expired or been revoked

## File Structure

```
ai_product_analyzer.py   # Main AI analysis module
app.py                   # Flask app with AI integration
image_analyzer.py        # Fallback CV-based analyzer
generate_description.py  # NLP description generation
templates/index.html     # Web interface
static/style.css         # Styling for AI features
```

## Example Usage

1. Start the app:
   ```bash
   python app.py
   ```

2. Open browser to `http://localhost:5000`

3. Click "Image Input"

4. Upload a product image (phone, shoe, coffee maker, etc.)

5. Click "Generate Description from Image"

6. See AI-powered analysis and realistic product description

## Next Steps

- Try with various product images
- Compare AI-powered results with traditional CV results
- Fine-tune product name or category if needed
- Copy description to clipboard for use in e-commerce sites

## Support

For API key issues:
- Claude: https://console.anthropic.com/
- OpenAI: https://platform.openai.com/account/api-keys

For questions about vision capabilities:
- Claude: https://docs.anthropic.com/claude/docs/vision
- OpenAI: https://platform.openai.com/docs/guides/vision
