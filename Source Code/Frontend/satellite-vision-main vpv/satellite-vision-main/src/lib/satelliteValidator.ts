// Satellite Image Validation Logic
// Validates if an uploaded image has characteristics of satellite/aerial imagery

export interface ValidationResult {
  isValid: boolean;
  confidence: number;
  message: string;
  details: string[];
}

// EuroSAT dataset characteristics:
// - 64x64 pixel patches (we accept similar sizes)
// - RGB satellite imagery from Sentinel-2
// - Land cover classification

export const EUROSAT_CLASSES = [
  { id: 'annual_crop', name: 'Annual Crop', color: 'hsl(42, 70%, 50%)', description: 'Agricultural fields with seasonal crops' },
  { id: 'forest', name: 'Forest', color: 'hsl(142, 76%, 36%)', description: 'Dense tree coverage areas' },
  { id: 'herbaceous_vegetation', name: 'Herbaceous Vegetation', color: 'hsl(90, 60%, 45%)', description: 'Grasslands and meadows' },
  { id: 'highway', name: 'Highway', color: 'hsl(0, 0%, 50%)', description: 'Major road infrastructure' },
  { id: 'industrial', name: 'Industrial', color: 'hsl(280, 50%, 45%)', description: 'Industrial facilities and zones' },
  { id: 'pasture', name: 'Pasture', color: 'hsl(80, 50%, 55%)', description: 'Grazing lands for livestock' },
  { id: 'permanent_crop', name: 'Permanent Crop', color: 'hsl(30, 65%, 45%)', description: 'Orchards and vineyards' },
  { id: 'residential', name: 'Residential', color: 'hsl(220, 60%, 55%)', description: 'Urban residential areas' },
  { id: 'river', name: 'River', color: 'hsl(199, 89%, 48%)', description: 'Water bodies and rivers' },
  { id: 'sea_lake', name: 'Sea/Lake', color: 'hsl(210, 80%, 40%)', description: 'Large water bodies' },
] as const;

export type EuroSATClass = typeof EUROSAT_CLASSES[number]['id'];

// Analyze image characteristics for satellite imagery detection (all EuroSAT classes)
export async function validateSatelliteImage(file: File): Promise<ValidationResult> {
  const details: string[] = [];
  let score = 0;
  
  // Check file type
  const validTypes = ['image/jpeg', 'image/png', 'image/tiff', 'image/webp'];
  if (!validTypes.includes(file.type)) {
    return {
      isValid: false,
      confidence: 0,
      message: 'Invalid file type',
      details: [`Expected image file (JPG, PNG, TIFF, WebP), got ${file.type || 'unknown'}`],
    };
  }
  details.push('✓ Valid image format');

  // Check image dimensions (accept any reasonable image size)
  try {
    const img = new Image();
    await new Promise((resolve, reject) => {
      img.onload = resolve;
      img.onerror = reject;
      img.src = URL.createObjectURL(file);
    });

    // Accept any image size as long as it's a valid image
    details.push(`✓ Image dimensions ${img.width}x${img.height} accepted`);
  } catch (error) {
    details.push('⚠ Could not verify image dimensions');
  }

  // Analyze image dimensions and color characteristics
  try {
    const imageAnalysis = await analyzeImageContent(file);
    
    // STRICT CHECK: Reject images with text/document characteristics
    if (imageAnalysis.hasTextCharacteristics) {
      return {
        isValid: false,
        confidence: 5,
        message: 'This appears to be a document or text-based image',
        details: [
          '✗ Image contains text/document patterns',
          '✗ Not a satellite image',
          '⚠ Please upload a satellite or aerial image',
        ],
      };
    }

    // STRICT CHECK: Reject images with too much white/uniform background
    if (imageAnalysis.hasDocumentBackground) {
      return {
        isValid: false,
        confidence: 10,
        message: 'This appears to be a document or screenshot',
        details: [
          '✗ Large uniform/white areas detected',
          '✗ Not a satellite image',
          '⚠ Please upload a satellite or aerial image',
        ],
      };
    }

    // STRICT CHECK: Reject images with artificial patterns
    if (imageAnalysis.hasArtificialPatterns) {
      return {
        isValid: false,
        confidence: 15,
        message: 'This appears to be an artificial image',
        details: [
          '✗ Artificial geometric patterns detected',
          '✗ Not a satellite image',
          '⚠ Please upload a satellite or aerial image',
        ],
      };
    }

    // SATELLITE CHECK: Must have characteristics of at least one EuroSAT class
    if (!imageAnalysis.hasSatelliteCharacteristics) {
      return {
        isValid: false,
        confidence: 20,
        message: 'This does not appear to be satellite imagery',
        details: [
          '✗ No recognizable land cover characteristics detected',
          '✗ Image lacks natural terrain features',
          '⚠ Please upload a satellite or aerial image showing natural landscapes',
        ],
      };
    }

    // Passed satellite check
    details.push('✓ Satellite imagery characteristics detected');
    score += 30;

    // Check dominant class characteristics
    const dominantClass = imageAnalysis.dominantClass;
    const dominantClassInfo = EUROSAT_CLASSES.find(c => c.id === dominantClass);

    if (dominantClassInfo) {
      details.push(`✓ Detected: ${dominantClassInfo.name} (${imageAnalysis.dominantClassConfidence.toFixed(1)}% confidence)`);
      score += 25;
    }

    // Check aspect ratio
    const aspectRatio = imageAnalysis.width / imageAnalysis.height;
    if (aspectRatio >= 0.8 && aspectRatio <= 1.25) {
      details.push('✓ Aspect ratio consistent with satellite patches');
      score += 15;
    } else if (aspectRatio >= 0.5 && aspectRatio <= 2) {
      details.push('⚠ Aspect ratio slightly unusual');
      score += 5;
    }

    // Check for natural texture
    if (imageAnalysis.hasNaturalTexture) {
      details.push('✓ Natural texture patterns detected');
      score += 20;
    } else {
      details.push('⚠ Texture patterns inconclusive');
      score += 5;
    }

    // Additional quality checks
    if (imageAnalysis.hasGoodQuality) {
      details.push('✓ Good image quality for classification');
      score += 10;
    }

  } catch (error) {
    details.push('⚠ Could not fully analyze image content');
    score += 10;
  }

  const confidence = Math.min(100, score);
  const isValid = confidence >= 50; // Lowered threshold since we accept all satellite classes

  return {
    isValid,
    confidence,
    message: isValid
      ? 'Valid satellite imagery detected'
      : 'Image does not appear to be satellite imagery. Please upload an image from the EuroSAT dataset.',
    details,
  };
}

interface ImageAnalysis {
  width: number;
  height: number;
  hasEarthyTones: boolean;
  hasNaturalTexture: boolean;
  hasSatelliteCharacteristics: boolean;
  hasTextCharacteristics: boolean;
  hasDocumentBackground: boolean;
  hasArtificialPatterns: boolean;
  hasGoodQuality: boolean;
  dominantClass: string;
  dominantClassConfidence: number;
  greenPercentage: number;
  bluePercentage: number;
  brownPercentage: number;
  grayPercentage: number;
}

async function analyzeImageContent(file: File): Promise<ImageAnalysis> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');

    img.onload = () => {
      if (!ctx) {
        reject(new Error('Could not get canvas context'));
        return;
      }

      // Use smaller canvas for analysis
      const maxSize = 200;
      const scale = Math.min(maxSize / img.width, maxSize / img.height, 1);
      canvas.width = img.width * scale;
      canvas.height = img.height * scale;

      ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
      const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
      const data = imageData.data;

      // Analyze color distribution
      let greenCount = 0;
      let blueCount = 0;
      let brownCount = 0;
      let grayCount = 0;
      let whiteCount = 0;
      let blackCount = 0;
      let highContrastCount = 0;
      const totalPixels = data.length / 4;
      let colorVariance = 0;
      let horizontalEdges = 0;
      let verticalEdges = 0;
      let prevR = 0, prevG = 0, prevB = 0;

      const width = canvas.width;
      const height = canvas.height;

      for (let i = 0; i < data.length; i += 4) {
        const r = data[i];
        const g = data[i + 1];
        const b = data[i + 2];
        const pixelIndex = i / 4;
        const x = pixelIndex % width;
        const y = Math.floor(pixelIndex / width);

        // Calculate color variance for texture detection
        if (i > 0) {
          const diff = Math.abs(r - prevR) + Math.abs(g - prevG) + Math.abs(b - prevB);
          colorVariance += diff;
          
          // Detect sharp edges (text/document characteristics)
          if (diff > 200) {
            highContrastCount++;
          }
        }
        prevR = r; prevG = g; prevB = b;

        // Detect horizontal and vertical edges (text patterns)
        if (x > 0 && y > 0) {
          const leftIndex = (y * width + (x - 1)) * 4;
          const topIndex = ((y - 1) * width + x) * 4;
          
          const leftDiff = Math.abs(r - data[leftIndex]) + Math.abs(g - data[leftIndex + 1]) + Math.abs(b - data[leftIndex + 2]);
          const topDiff = Math.abs(r - data[topIndex]) + Math.abs(g - data[topIndex + 1]) + Math.abs(b - data[topIndex + 2]);
          
          if (leftDiff > 150) verticalEdges++;
          if (topDiff > 150) horizontalEdges++;
        }

        // White detection (documents, screenshots)
        if (r > 230 && g > 230 && b > 230) {
          whiteCount++;
        }
        
        // Black detection (text)
        if (r < 30 && g < 30 && b < 30) {
          blackCount++;
        }

        // ENHANCED FOREST DETECTION - Multiple green detection methods
        
        // Method 1: Standard vegetation (green dominant)
        if (g > r * 0.9 && g > b * 1.1) greenCount++;
        
        // Method 2: Dark forest greens (lower brightness forest canopy)
        const brightness = (r + g + b) / 3;
        if (g > r && g > b && brightness < 120 && brightness > 30) greenCount++;
        
        // Method 3: Light vegetation (bright greens, meadows)
        if (g > 80 && g > r * 1.05 && g > b * 1.05) greenCount++;
        
        // Method 4: HSL-based green detection for better accuracy
        const max = Math.max(r, g, b);
        const min = Math.min(r, g, b);
        const lightness = (max + min) / 2;
        if (max !== min) {
          const saturation = lightness > 127 
            ? (max - min) / (510 - max - min) 
            : (max - min) / (max + min);
          // Green hue range (approximately 60-180 degrees)
          if (g === max && saturation > 0.15 && lightness > 20 && lightness < 200) {
            const hue = 60 + ((b - r) / (max - min)) * 60;
            if (hue >= 60 && hue <= 180) greenCount++;
          }
        }
        
        // Other terrain types
        if (b > r * 1.1 && b > g * 1.1) blueCount++; // Water
        if (r > b * 1.1 && g > b * 0.9 && Math.abs(r - g) < 60) brownCount++; // Soil/sand
        if (Math.abs(r - g) < 30 && Math.abs(g - b) < 30 && r > 50 && r < 200) grayCount++; // Urban/roads
      }

      const earthyRatio = (greenCount + blueCount + brownCount + grayCount) / totalPixels;
      const whiteRatio = whiteCount / totalPixels;
      const blackRatio = blackCount / totalPixels;
      const avgVariance = colorVariance / totalPixels;
      const edgeRatio = (horizontalEdges + verticalEdges) / totalPixels;
      const highContrastRatio = highContrastCount / totalPixels;

      // Text characteristics: high contrast edges, mix of black and white
      const hasTextCharacteristics = 
        (blackRatio > 0.05 && whiteRatio > 0.3) || // Black text on white background
        (highContrastRatio > 0.15 && edgeRatio > 0.1) || // High contrast with many edges
        (horizontalEdges > totalPixels * 0.08 && verticalEdges > totalPixels * 0.03); // Text-like edge patterns

      // Document background: large white/uniform areas
      const hasDocumentBackground = 
        whiteRatio > 0.4 || // Mostly white
        (whiteRatio > 0.25 && avgVariance < 20); // Significant white with low variance

      // Artificial patterns: regular geometric edges
      const hasArtificialPatterns = 
        (edgeRatio > 0.15 && avgVariance > 100) || // Many sharp edges
        (highContrastRatio > 0.2); // Very high contrast throughout

      // Calculate percentages
      const adjustedGreenCount = greenCount / 4;
      const greenPercentage = (adjustedGreenCount / totalPixels) * 100;
      const bluePercentage = (blueCount / totalPixels) * 100;
      const brownPercentage = (brownCount / totalPixels) * 100;
      const grayPercentage = (grayCount / totalPixels) * 100;

      // Determine dominant class based on analysis
      const colorDistribution: Record<string, number> = {
        forest: greenPercentage * 0.4, // Dense forest
        herbaceous_vegetation: greenPercentage * 0.3, // Light vegetation
        river: bluePercentage * (avgVariance < 30 ? 1.5 : 0.5),
        sea_lake: bluePercentage * (avgVariance < 20 ? 1.5 : 0.5),
        annual_crop: brownPercentage * 0.8,
        permanent_crop: brownPercentage * 0.6,
        pasture: greenPercentage * 0.2,
        industrial: grayPercentage * (avgVariance > 40 ? 1.3 : 0.7),
        residential: grayPercentage * (avgVariance > 30 ? 1.2 : 0.8),
        highway: grayPercentage * (avgVariance < 25 ? 1.4 : 0.6),
      };

      // Find dominant class
      let dominantClass = 'forest';
      let maxScore = 0;
      for (const [cls, score] of Object.entries(colorDistribution)) {
        if (score > maxScore) {
          maxScore = score;
          dominantClass = cls;
        }
      }

      // Determine confidence based on how dominant the top class is
      const dominantClassConfidence = maxScore;

      // Quality check: not too blurry or noisy
      const hasGoodQuality = avgVariance > 5 && avgVariance < 150 && !hasTextCharacteristics && !hasDocumentBackground;

      resolve({
        width: img.width,
        height: img.height,
        hasEarthyTones: earthyRatio > 0.35,
        hasNaturalTexture: avgVariance > 15 && avgVariance < 120,
        hasSatelliteCharacteristics: earthyRatio > 0.3 && avgVariance > 8 && avgVariance < 130 && !hasTextCharacteristics && !hasDocumentBackground,
        hasTextCharacteristics,
        hasDocumentBackground,
        hasArtificialPatterns,
        hasGoodQuality,
        dominantClass,
        dominantClassConfidence,
        greenPercentage,
        bluePercentage,
        brownPercentage,
        grayPercentage,
      });
    };

    img.onerror = () => reject(new Error('Failed to load image'));
    img.src = URL.createObjectURL(file);
  });
}

// Classification result based on actual image analysis
export interface ClassificationResult {
  predictions: Array<{
    class: typeof EUROSAT_CLASSES[number];
    probability: number;
  }>;
  processingTime: number;
  imageSize: { width: number; height: number };
  analysisDetails: {
    greenCoverage: number;
    dominantFeature: string;
    confidence: 'high' | 'medium' | 'low';
  };
}

// Analyze image and provide real predictions based on color/texture analysis
export async function classifyImage(file: File): Promise<ClassificationResult> {
  const startTime = performance.now();

  // Perform actual image analysis
  const analysis = await analyzeImageForClassification(file);
  const processingTime = performance.now() - startTime;

  // Generate predictions based on actual image content
  const predictions = generatePredictionsFromAnalysis(analysis);

  return {
    predictions,
    processingTime,
    imageSize: { width: analysis.width, height: analysis.height },
    analysisDetails: {
      greenCoverage: analysis.greenPercentage,
      dominantFeature: analysis.dominantClass,
      confidence: analysis.confidence,
    },
  };
}

// Batch classification for multiple images
export async function classifyImages(images: Array<{ file: File; preview: string }>): Promise<Array<{ result: ClassificationResult; file: File; preview: string }>> {
  const results: Array<{ result: ClassificationResult; file: File; preview: string }> = [];

  for (const image of images) {
    try {
      const result = await classifyImage(image.file);
      results.push({ result, file: image.file, preview: image.preview });
    } catch (error) {
      console.error(`Failed to classify image:`, error);
      // Create a fallback result for failed classifications
      const fallbackResult: ClassificationResult = {
        predictions: [],
        imageSize: { width: 0, height: 0 },
        processingTime: 0,
        analysisDetails: {
          greenCoverage: 0,
          dominantFeature: 'unknown',
          confidence: 'low'
        }
      };
      results.push({ result: fallbackResult, file: image.file, preview: image.preview });
    }
  }

  return results;
}

interface DetailedImageAnalysis {
  width: number;
  height: number;
  greenPercentage: number;
  bluePercentage: number;
  brownPercentage: number;
  grayPercentage: number;
  dominantClass: string;
  confidence: 'high' | 'medium' | 'low';
  textureScore: number;
  colorDistribution: Record<string, number>;
}

async function analyzeImageForClassification(file: File): Promise<DetailedImageAnalysis> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');

    img.onload = () => {
      if (!ctx) {
        reject(new Error('Could not get canvas context'));
        return;
      }

      // Use larger canvas for better accuracy
      const maxSize = 256;
      const scale = Math.min(maxSize / img.width, maxSize / img.height, 1);
      canvas.width = img.width * scale;
      canvas.height = img.height * scale;

      ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
      const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
      const data = imageData.data;

      const totalPixels = data.length / 4;
      
      // Color category counters
      let forestGreen = 0;      // Dense forest
      let lightGreen = 0;       // Herbaceous/pasture
      let darkGreen = 0;        // Dark canopy
      let waterBlue = 0;        // River/sea
      let brownEarth = 0;       // Crops/soil
      let urbanGray = 0;        // Industrial/residential/highway
      let yellowCrop = 0;       // Annual crops
      
      let textureVariance = 0;
      const horizontalEdges = 0;
      const verticalEdges = 0;
      const highContrastCount = 0;
      let prevR = 0, prevG = 0, prevB = 0;

      const width = canvas.width;
      const height = canvas.height;

      for (let i = 0; i < data.length; i += 4) {
        const r = data[i];
        const g = data[i + 1];
        const b = data[i + 2];

        // Texture analysis
        if (i > 0) {
          textureVariance += Math.abs(r - prevR) + Math.abs(g - prevG) + Math.abs(b - prevB);
        }
        prevR = r; prevG = g; prevB = b;

        const brightness = (r + g + b) / 3;
        const max = Math.max(r, g, b);
        const min = Math.min(r, g, b);
        const saturation = max === 0 ? 0 : (max - min) / max;

        // FOREST DETECTION - Multiple green detection methods
        // Dark forest canopy (shadows in dense forest)
        if (g > r && g > b && brightness < 80 && brightness > 20 && saturation > 0.15) {
          darkGreen++;
          forestGreen++;
        }
        // Dense forest (medium brightness, green dominant)
        else if (g > r * 1.1 && g > b * 1.15 && brightness >= 40 && brightness < 130 && saturation > 0.2) {
          forestGreen++;
        }
        // Additional forest detection (broader green range for forest images)
        else if (g > r * 1.05 && g > b * 1.1 && brightness >= 30 && brightness < 100 && saturation > 0.1) {
          forestGreen++;
        }
        // Light vegetation / herbaceous
        else if (g > r * 1.05 && g > b * 1.05 && brightness >= 100 && brightness < 180) {
          lightGreen++;
        }
        // Water bodies (blue dominant)
        else if (b > r * 1.15 && b > g * 1.1 && saturation > 0.15) {
          waterBlue++;
        }
        // Crops / agricultural (yellow-brown tones)
        else if (r > b * 1.2 && g > b * 1.1 && Math.abs(r - g) < 50 && brightness > 100) {
          yellowCrop++;
          brownEarth++;
        }
        // Soil / permanent crops (brown tones)
        else if (r > b * 1.1 && g > b * 0.9 && Math.abs(r - g) < 40 && brightness < 140) {
          brownEarth++;
        }
        // Urban / industrial (gray tones with low saturation)
        else if (saturation < 0.15 && brightness > 60 && brightness < 200) {
          urbanGray++;
        }
      }

      // Calculate percentages
      const greenPercentage = ((forestGreen + lightGreen + darkGreen) / totalPixels) * 100;
      const bluePercentage = (waterBlue / totalPixels) * 100;
      const brownPercentage = (brownEarth / totalPixels) * 100;
      const grayPercentage = (urbanGray / totalPixels) * 100;
      const avgTextureVariance = textureVariance / totalPixels;
      const highContrastRatio = highContrastCount / totalPixels;

      // Enhanced classification using color, texture, and pattern analysis
      const colorDistribution: Record<string, number> = {
        // Natural vegetation classes
        forest: (forestGreen / totalPixels) * 100 * (avgTextureVariance > 20 ? 1.4 : 1.0), // Dense forest with texture
        herbaceous_vegetation: (lightGreen / totalPixels) * 100 * (avgTextureVariance > 15 ? 1.2 : 0.8), // Grasslands
        pasture: (lightGreen / totalPixels) * 100 * (avgTextureVariance < 25 ? 1.1 : 0.9), // Managed grasslands

        // Agricultural classes
        annual_crop: (yellowCrop / totalPixels) * 100 * (avgTextureVariance > 10 ? 1.3 : 0.7), // Seasonal crops
        permanent_crop: (brownEarth / totalPixels) * 100 * (avgTextureVariance > 12 ? 1.2 : 0.8), // Orchards/vineyards

        // Water classes
        river: (waterBlue / totalPixels) * 100 * (avgTextureVariance > 25 && avgTextureVariance < 50 ? 1.5 : 0.6), // Rivers with flow texture
        sea_lake: (waterBlue / totalPixels) * 100 * (avgTextureVariance < 20 ? 1.4 : 0.7), // Calm water bodies

        // Urban classes
        highway: (urbanGray / totalPixels) * 100 * (avgTextureVariance < 20 && horizontalEdges > verticalEdges * 1.5 ? 1.6 : 0.5), // Roads with linear patterns
        industrial: (urbanGray / totalPixels) * 100 * (avgTextureVariance > 40 && highContrastRatio > 0.1 ? 1.4 : 0.8), // Factories with high activity
        residential: (urbanGray / totalPixels) * 100 * (avgTextureVariance > 25 && avgTextureVariance < 40 ? 1.3 : 0.9), // Urban areas
      };

      // Find dominant class
      let dominantClass = 'forest';
      let maxScore = 0;
      for (const [cls, score] of Object.entries(colorDistribution)) {
        if (score > maxScore) {
          maxScore = score;
          dominantClass = cls;
        }
      }

      // Determine confidence based on how dominant the top class is
      let confidence: 'high' | 'medium' | 'low' = 'low';
      if (maxScore > 25) confidence = 'high';
      else if (maxScore > 15) confidence = 'medium';

      resolve({
        width: img.width,
        height: img.height,
        greenPercentage,
        bluePercentage,
        brownPercentage,
        grayPercentage,
        dominantClass,
        confidence,
        textureScore: avgTextureVariance,
        colorDistribution,
      });
    };

    img.onerror = () => reject(new Error('Failed to load image'));
    img.src = URL.createObjectURL(file);
  });
}

function generatePredictionsFromAnalysis(analysis: DetailedImageAnalysis): ClassificationResult['predictions'] {
  const predictions: ClassificationResult['predictions'] = [];
  
  // Sort classes by their score from color distribution
  const sortedClasses = Object.entries(analysis.colorDistribution)
    .sort(([, a], [, b]) => b - a)
    .slice(0, 5);

  // Calculate total for normalization
  const total = sortedClasses.reduce((sum, [, score]) => sum + Math.max(score, 1), 0);

  for (const [classId, score] of sortedClasses) {
    const euroClass = EUROSAT_CLASSES.find(c => c.id === classId);
    if (euroClass) {
      // Normalize probability and apply confidence boost to dominant class
      let probability = Math.max(score, 1) / total;
      
      // Boost the top prediction if confidence is high
      if (classId === analysis.dominantClass && analysis.confidence === 'high') {
        probability = Math.min(probability * 1.3, 0.95);
      }
      
      predictions.push({
        class: euroClass,
        probability,
      });
    }
  }

  // Normalize to sum to 1
  const probTotal = predictions.reduce((sum, p) => sum + p.probability, 0);
  predictions.forEach(p => p.probability = p.probability / probTotal);

  // Sort by probability descending
  predictions.sort((a, b) => b.probability - a.probability);

  return predictions;
}
