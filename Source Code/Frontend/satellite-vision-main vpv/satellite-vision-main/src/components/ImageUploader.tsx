import { useState, useCallback, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Upload, X, AlertCircle, CheckCircle2, Loader2, Satellite } from 'lucide-react';
import { validateSatelliteImage, type ValidationResult } from '@/lib/satelliteValidator';

interface ImageUploaderProps {
  onImagesValidated: (images: Array<{ file: File; preview: string }>) => void;
  onValidationFailed: (results: Array<{ file: File; result: ValidationResult }>) => void;
  isProcessing?: boolean;
}

export function ImageUploader({ onImagesValidated, onValidationFailed, isProcessing }: ImageUploaderProps) {
  const [isDragging, setIsDragging] = useState(false);
  const [isValidating, setIsValidating] = useState(false);
  const [validationResults, setValidationResults] = useState<Array<{ file: File; result: ValidationResult; preview: string }>>([]);
  const [previewUrls, setPreviewUrls] = useState<Array<{ file: File; url: string }>>([]);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFiles = useCallback(async (files: FileList | File[]) => {
    setIsValidating(true);
    setValidationResults([]);
    setPreviewUrls([]);

    const fileArray = Array.from(files);
    const validImages: Array<{ file: File; preview: string }> = [];
    const failedValidations: Array<{ file: File; result: ValidationResult }> = [];
    const newPreviews: Array<{ file: File; url: string }> = [];
    const newResults: Array<{ file: File; result: ValidationResult; preview: string }> = [];

    for (const file of fileArray) {
      // Create preview
      const preview = URL.createObjectURL(file);
      newPreviews.push({ file, url: preview });

      // Validate the image
      const result = await validateSatelliteImage(file);
      newResults.push({ file, result, preview });

      if (result.isValid) {
        validImages.push({ file, preview });
      } else {
        failedValidations.push({ file, result });
      }
    }

    setPreviewUrls(newPreviews);
    setValidationResults(newResults);
    setIsValidating(false);

    if (validImages.length > 0) {
      onImagesValidated(validImages);
    }
    if (failedValidations.length > 0) {
      onValidationFailed(failedValidations);
    }
  }, [onImagesValidated, onValidationFailed]);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);

    const files = e.dataTransfer.files;
    if (files && files.length > 0) {
      handleFiles(files);
    }
  }, [handleFiles]);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  }, []);

  const handleClick = () => {
    fileInputRef.current?.click();
  };

  const handleFileInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (files && files.length > 0) {
      handleFiles(files);
    }
  };

  const clearImages = () => {
    setPreviewUrls([]);
    setValidationResults([]);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const getZoneClass = () => {
    const hasValid = validationResults.some(v => v.result.isValid);
    const hasInvalid = validationResults.some(v => !v.result.isValid);
    if (hasValid && !hasInvalid) return 'valid';
    if (hasInvalid) return 'invalid';
    if (isDragging) return 'drag-over';
    return '';
  };

  return (
    <div className="w-full">
      <input
        ref={fileInputRef}
        type="file"
        accept="image/*"
        multiple
        onChange={handleFileInput}
        className="hidden"
      />

      <motion.div
        className={`upload-zone relative overflow-hidden rounded-xl p-8 cursor-pointer min-h-[300px] flex flex-col items-center justify-center ${getZoneClass()}`}
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onClick={!previewUrls.length ? handleClick : undefined}
        whileHover={!previewUrls.length ? { scale: 1.01 } : {}}
        whileTap={!previewUrls.length ? { scale: 0.99 } : {}}
      >
        {/* Scanning effect when validating */}
        {isValidating && (
          <div className="scan-line absolute inset-0 pointer-events-none" />
        )}

        <AnimatePresence mode="wait">
          {!previewUrls.length && !isValidating && (
            <motion.div
              key="upload-prompt"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              className="flex flex-col items-center text-center"
            >
              <motion.div
                className="w-20 h-20 rounded-full bg-muted flex items-center justify-center mb-6"
                animate={{ y: [0, -8, 0] }}
                transition={{ duration: 2, repeat: Infinity, ease: "easeInOut" }}
              >
                <Satellite className="w-10 h-10 text-primary" />
              </motion.div>
              <h3 className="text-xl font-semibold mb-2">
                {isDragging ? 'Drop your satellite images' : 'Upload Satellite Images'}
              </h3>
              <p className="text-muted-foreground mb-4 max-w-sm">
                Drag and drop satellite or aerial images, or click to browse.
                Multiple images supported. Only satellite imagery will be accepted.
              </p>
              <div className="flex items-center gap-2 text-sm text-muted-foreground">
                <Upload className="w-4 h-4" />
                <span>Supports JPG, PNG, TIFF, WebP</span>
              </div>
            </motion.div>
          )}

          {isValidating && (
            <motion.div
              key="validating"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="flex flex-col items-center text-center"
            >
              <div className="relative mb-6">
                {previewUrls.length > 0 && (
                  <div className="flex gap-2">
                    {previewUrls.slice(0, 3).map((preview, index) => (
                      <img
                        key={index}
                        src={preview.url}
                        alt={`Uploading ${index + 1}`}
                        className="w-20 h-20 object-cover rounded-lg opacity-50"
                      />
                    ))}
                    {previewUrls.length > 3 && (
                      <div className="w-20 h-20 bg-muted rounded-lg flex items-center justify-center text-sm text-muted-foreground">
                        +{previewUrls.length - 3}
                      </div>
                    )}
                  </div>
                )}
                <div className="absolute inset-0 flex items-center justify-center">
                  <Loader2 className="w-12 h-12 text-primary animate-spin" />
                </div>
              </div>
              <h3 className="text-xl font-semibold mb-2">Validating Images...</h3>
              <p className="text-muted-foreground">
                Analyzing {previewUrls.length} image{previewUrls.length !== 1 ? 's' : ''} characteristics
              </p>
            </motion.div>
          )}

          {previewUrls.length > 0 && validationResults.length > 0 && !isValidating && (
            <motion.div
              key="results"
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.9 }}
              className="w-full"
            >
              <div className="space-y-4">
                <h3 className="text-lg font-semibold text-center">
                  {validationResults.length} Image{validationResults.length !== 1 ? 's' : ''} Processed
                </h3>

                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                  {validationResults.map((item, index) => (
                    <motion.div
                      key={index}
                      initial={{ opacity: 0, y: 20 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ delay: index * 0.1 }}
                      className="bg-card rounded-lg p-4 border border-border"
                    >
                      {/* Image Preview */}
                      <div className="relative mb-3">
                        <img
                          src={item.preview}
                          alt={`Preview ${index + 1}`}
                          className="w-full h-32 object-cover rounded-lg"
                        />
                        <div className="absolute top-2 right-2">
                          {item.result.isValid ? (
                            <CheckCircle2 className="w-6 h-6 text-success bg-white rounded-full p-1" />
                          ) : (
                            <AlertCircle className="w-6 h-6 text-destructive bg-white rounded-full p-1" />
                          )}
                        </div>
                      </div>

                      {/* Validation Details */}
                      <div className="space-y-2">
                        <h4 className={`text-sm font-medium ${item.result.isValid ? 'text-success' : 'text-destructive'}`}>
                          {item.result.message}
                        </h4>

                        <div className="flex justify-between text-xs">
                          <span className="text-muted-foreground">Confidence</span>
                          <span className="font-mono">{item.result.confidence}%</span>
                        </div>

                        <div className="h-1.5 bg-muted rounded-full overflow-hidden">
                          <motion.div
                            className={`h-full rounded-full ${item.result.isValid ? 'bg-gradient-success' : 'bg-destructive'}`}
                            initial={{ width: 0 }}
                            animate={{ width: `${item.result.confidence}%` }}
                            transition={{ duration: 1, delay: index * 0.1, ease: "easeOut" }}
                          />
                        </div>
                      </div>
                    </motion.div>
                  ))}
                </div>

                <div className="flex justify-center gap-4">
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      clearImages();
                    }}
                    className="px-6 py-2 bg-muted hover:bg-muted/80 rounded-lg font-medium transition-colors"
                  >
                    Clear All
                  </button>

                  {validationResults.some(v => v.result.isValid) && (
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        // This will be handled by parent component
                      }}
                      className="px-6 py-2 bg-primary text-primary-foreground rounded-lg hover:bg-primary/90 transition-colors"
                    >
                      Classify Valid Images
                    </button>
                  )}
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Processing overlay */}
        {isProcessing && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="absolute inset-0 bg-background/80 backdrop-blur-sm flex flex-col items-center justify-center"
          >
            <Loader2 className="w-12 h-12 text-primary animate-spin mb-4" />
            <p className="text-lg font-medium">Classifying land cover...</p>
            <p className="text-sm text-muted-foreground">Running AI model</p>
          </motion.div>
        )}
      </motion.div>
    </div>
  );
}
