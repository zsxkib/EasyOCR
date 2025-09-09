#!/usr/bin/env python3
"""
Test script for the deployed EasyOCR model on Replicate
"""

import replicate
import json

def test_ocr_model():
    print("🚀 Testing EasyOCR model on Replicate...")
    
    # Test with the provided image URL
    image_url = "https://replicate.delivery/pbxt/NcEiSBJPolWph3PlpDFJIvVMUlSW8sboi9rz9O1ymUE2kRBG/image.png"
    
    try:
        output = replicate.run(
            "zsxkib/easyocr:d12cc208b912320422d40d788b3490adace66cf1a147bd9fa070285aa3a22996",
            input={
                "image": image_url,
                "languages": "",
                "text_only": False,
                "preprocessing": True,
                "include_bboxes": True,
                "min_confidence": 0.25,
                "include_polygons": False
            }
        )
        
        print("✅ Model run completed successfully!")
        print(f"📄 Markdown file: {output['markdown']}")
        print(f"📊 Metadata preview:")
        
        # Parse the metadata JSON
        metadata = json.loads(output["metadata"])
        print(f"   - Total regions: {metadata.get('total_regions', 'N/A')}")
        print(f"   - Average confidence: {metadata.get('avg_confidence', 'N/A')}")
        print(f"   - Languages used: {metadata.get('languages_used', 'N/A')}")
        
        # Save the full metadata to a file for the rendering script
        with open("ocr_output.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        
        print("💾 Saved full metadata to 'ocr_output.json'")
        
        # Print first few text regions
        regions = metadata.get('regions', [])
        if regions:
            print(f"📝 First {min(3, len(regions))} text regions:")
            for i, region in enumerate(regions[:3]):
                print(f"   {i+1}. '{region.get('text', '')}' (confidence: {region.get('confidence', 0):.3f})")
                if region.get('x1') is not None:
                    print(f"      Bbox: ({region['x1']}, {region['y1']}) -> ({region['x2']}, {region['y2']})")
        
        return metadata
        
    except Exception as e:
        print(f"❌ Error running model: {e}")
        return None

if __name__ == "__main__":
    result = test_ocr_model()
    if result:
        print("🎉 Test completed successfully!")
    else:
        print("💥 Test failed!")
