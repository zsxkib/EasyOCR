# 🚀 Production Deployment Checklist

## Pre-Deployment Configuration

### 1. **Environment Configuration**
- [ ] **Update `cog.yaml`:**
  ```yaml
  gpu: true  # Change from false to true
  cuda: "12.1"  # Verify CUDA version compatibility
  ```

- [ ] **Verify system packages:**
  - [ ] All OpenGL libraries included
  - [ ] CUDA-compatible packages listed
  - [ ] Memory-optimized configurations

- [ ] **Check `requirements.txt`:**
  - [ ] All versions pinned for reproducibility
  - [ ] GPU-compatible PyTorch version (2.1.0)
  - [ ] EasyOCR version locked (1.7.1)

### 2. **Code Validation**
- [ ] **Test locally on Mac** (CPU fallback):
  ```bash
  cog predict -i image=@test_screenshot.png
  ```
- [ ] **Verify preprocessing pipeline works without GPU**
- [ ] **Confirm error handling for edge cases**
- [ ] **Test multi-language functionality**

### 3. **Model Preparation**
- [ ] **Pre-download models** (built into Docker during build):
  - English base model
  - European language pack (es, fr, de)
  - Asian language pack (ja, ko, zh)
- [ ] **Verify model warmup** in setup() method
- [ ] **Test model loading time** (should be <10s cold start)

## Performance Testing

### 4. **Local Performance Validation**
- [ ] **Test various image sizes:**
  - [ ] Mobile screenshots (750x1334)
  - [ ] Desktop screenshots (1920x1080)
  - [ ] High-res captures (3840x2160)
- [ ] **Measure processing times** (CPU baseline)
- [ ] **Verify memory usage** stays within limits
- [ ] **Test confidence threshold behavior**

### 5. **Input Format Validation**
- [ ] **Test supported formats:**
  - [ ] PNG (recommended for screenshots)
  - [ ] JPEG (photos and compressed images)
  - [ ] WebP (modern web format)
  - [ ] TIFF (high-quality scans)
- [ ] **Test edge cases:**
  - [ ] Very small images (<100px)
  - [ ] Large images (approaching 20MB limit)
  - [ ] Corrupted/invalid files
  - [ ] Empty/blank images

## Production Deployment

### 6. **Build & Deploy**
- [ ] **Build production image:**
  ```bash
  cog build
  ```
- [ ] **Push to Replicate:**
  ```bash
  cog push r8.im/your-username/screenshot-ocr
  ```
- [ ] **Verify successful deployment** on Replicate dashboard

### 7. **Production Testing**
- [ ] **Test GPU acceleration** (should be 5-10x faster than Mac CPU)
- [ ] **Verify cold start performance** (<10 seconds)
- [ ] **Test warm request latency** (200-500ms target)
- [ ] **Validate concurrent request handling**

### 8. **Error Handling Verification**
- [ ] **Test invalid inputs:**
  - [ ] Non-existent file paths
  - [ ] Unsupported image formats
  - [ ] Oversized files (>20MB)
  - [ ] Invalid language codes
- [ ] **Verify graceful error responses**
- [ ] **Check error logging functionality**

## Monitoring & Performance

### 9. **Performance Benchmarks**
Expected performance on NVIDIA L40s:

| Metric | Target | Actual | Status |
|--------|--------|--------|---------|
| Cold start | <10s | ___ | ⏳ |
| 1080p processing | 200-500ms | ___ | ⏳ |
| 4K processing | 400-800ms | ___ | ⏳ |
| Memory usage | <8GB | ___ | ⏳ |
| Concurrent requests | 10+ | ___ | ⏳ |

### 10. **Monitoring Setup**
- [ ] **Response time monitoring**
- [ ] **Error rate tracking**
- [ ] **Memory usage alerts**
- [ ] **GPU utilization monitoring**
- [ ] **Request volume tracking**

### 11. **Load Testing**
- [ ] **Test autoscaling behavior**
- [ ] **Verify request queuing under load**
- [ ] **Test memory cleanup between requests**
- [ ] **Validate performance under sustained load**

## Post-Deployment Validation

### 12. **Functional Testing**
- [ ] **End-to-end API testing:**
  ```bash
  curl -X POST https://api.replicate.com/v1/predictions \
    -H "Authorization: Token YOUR_TOKEN" \
    -H "Content-Type: application/json" \
    -d '{
      "version": "YOUR_MODEL_VERSION",
      "input": {
        "image": "data:image/png;base64,BASE64_IMAGE",
        "languages": "en",
        "min_confidence": 0.25,
        "preprocessing": true
      }
    }'
  ```

- [ ] **Test response format validation**
- [ ] **Verify bounding box accuracy**
- [ ] **Confirm confidence score reliability**

### 13. **Integration Testing**
- [ ] **Test with production image types:**
  - [ ] Web screenshots
  - [ ] Mobile app screenshots
  - [ ] Document scans
  - [ ] Mixed-language content
- [ ] **Validate against known benchmarks**
- [ ] **Compare accuracy with baseline models**

### 14. **Documentation Verification**
- [ ] **API documentation matches implementation**
- [ ] **Example code works correctly**
- [ ] **Error codes documented properly**
- [ ] **Performance expectations accurate**

## Rollback Plan

### 15. **Contingency Preparation**
- [ ] **Previous version tagged and ready**
- [ ] **Rollback procedure documented**
- [ ] **Monitoring alerts configured**
- [ ] **Emergency contacts identified**

---

## Quick Test Commands

### Local Development Test:
```bash
# Test basic functionality
cog predict -i image=@test_screenshot.png

# Test with specific parameters
cog predict -i image=@test_screenshot.png -i languages="en,es" -i min_confidence=0.3

# Test preprocessing disabled
cog predict -i image=@test_screenshot.png -i preprocessing=false
```

### Production API Test:
```python
import replicate

model = replicate.models.get("your-username/screenshot-ocr")
prediction = model.predict(
    image="https://example.com/screenshot.png",
    languages="en",
    min_confidence=0.25,
    preprocessing=True
)
print(prediction)
```

---

## Performance Expectations

| Environment | Processing Time | Memory | Throughput |
|-------------|-----------------|---------|------------|
| **Mac Development** (CPU) | 2-5 seconds | ~2GB | 1-2 req/min |
| **Production** (GPU) | 200-500ms | ~6GB | 100+ req/min |

---

## Support & Troubleshooting

If deployment issues occur:

1. **Check logs** for specific error messages
2. **Verify CUDA compatibility** (12.1 required)
3. **Ensure adequate memory** (8GB+ recommended)
4. **Test with minimal configuration** first
5. **Rollback if critical issues** persist

---

*Complete this checklist before deploying to production. Each checked item represents a verified, production-ready component.*
